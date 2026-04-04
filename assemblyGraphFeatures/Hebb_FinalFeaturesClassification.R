
library(tidyverse)


dat <- read.csv('graph_features_ALL_subjects.csv')

unique(dat$activity_type)
unique(dat$threshold_pct)
unique(dat$max_iai_trs)
unique(dat$act_count_pct)
unique(dat$sex)





# ==============================================================================
# SVM with 5-Fold Nested CV — classify sex
# ==============================================================================

required_pkgs <- c("e1071", "caret", "pROC", "dplyr")
invisible(lapply(required_pkgs, function(p) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  library(p, character.only = TRUE)
}))

FEATURES <- c("activation_count", "Nodes", "L1", "RE", "PE",
              "L2", "L3", "LCC", "LSC", "ATD",
              "Density", "Diameter", "ASP", "CC")
TARGET          <- "sex"
MIN_N_PER_CLASS <- 250

tune_grid <- expand.grid(
  cost  = c(0.01, 0.1, 1, 10, 100),
  gamma = c(0.001, 0.01, 0.1, 1)
)

# ── Metrics function ──────────────────────────────────────────────────────────
compute_metrics <- function(actual, predicted_prob, predicted_class, fold_id,
                            best_cost, best_gamma) {
  cm  <- confusionMatrix(as.factor(predicted_class),
                         as.factor(actual), positive = "1")
  roc_obj <- tryCatch(
    pROC::roc(as.numeric(as.character(actual)),
              as.numeric(predicted_prob), quiet = TRUE),
    error = function(e) NULL
  )
  auc_val <- if (!is.null(roc_obj)) as.numeric(pROC::auc(roc_obj)) else NA
  
  cm_table <- cm$table
  TN <- cm_table["0", "0"]
  FP <- cm_table["1", "0"]
  FN <- cm_table["0", "1"]
  TP <- cm_table["1", "1"]
  
  data.frame(
    outer_fold   = fold_id,
    best_cost    = best_cost,
    best_gamma   = best_gamma,
    TP = TP, TN = TN, FP = FP, FN = FN,
    accuracy     = cm$overall["Accuracy"],
    sensitivity  = cm$byClass["Sensitivity"],
    specificity  = cm$byClass["Specificity"],
    ppv          = cm$byClass["Pos Pred Value"],
    npv          = cm$byClass["Neg Pred Value"],
    f1           = cm$byClass["F1"],
    balanced_acc = cm$byClass["Balanced Accuracy"],
    auc          = auc_val,
    n_test       = length(actual),
    row.names    = NULL
  )
}

# ── Permutation feature importance ───────────────────────────────────────────
svm_feature_importance_rbf <- function(model, X_test, y_test, n_perm = 20) {
  base_acc <- mean(predict(model, X_test) == y_test)
  importance <- sapply(colnames(X_test), function(feat) {
    perm_accs <- replicate(n_perm, {
      X_perm         <- X_test
      X_perm[[feat]] <- sample(X_perm[[feat]])
      mean(predict(model, X_perm) == y_test)
    })
    base_acc - mean(perm_accs)
  })
  sort(importance, decreasing = TRUE)
}

# ── Combos ────────────────────────────────────────────────────────────────────
combos <- expand.grid(
  activity_type = unique(dat$activity_type),
  threshold_pct = unique(dat$threshold_pct),
  max_iai_trs   = unique(dat$max_iai_trs),
  act_count_pct = unique(dat$act_count_pct),
  stringsAsFactors = FALSE
)

all_metrics      <- list()
all_features     <- list()
skipped_versions <- data.frame()

set.seed(42)

for (i in seq_len(nrow(combos))) {
  
  act  <- combos$activity_type[i]
  thr  <- combos$threshold_pct[i]
  miai <- combos$max_iai_trs[i]
  acp  <- combos$act_count_pct[i]
  
  version_tag <- sprintf("act=%s_thr=%s_miai=%s_acp=%s", act, thr, miai, acp)
  cat("\n[", i, "/", nrow(combos), "]", version_tag, "\n")
  
  sub <- dat %>%
    filter(activity_type == act, threshold_pct == thr,
           max_iai_trs == miai, act_count_pct == acp) %>%
    select(all_of(c(FEATURES, TARGET))) %>%
    filter(complete.cases(.))
  
  # ── Class size check ────────────────────────────────────────────────────────
  class_counts <- table(sub[[TARGET]])
  min_class_n  <- min(class_counts)
  
  if (min_class_n < MIN_N_PER_CLASS) {
    cat(sprintf("SKIP: %s — min sex group n=%d. Counts: %s\n",
                version_tag, min_class_n,
                paste(names(class_counts), class_counts, sep="=", collapse=", ")))
    skipped_versions <- rbind(skipped_versions, data.frame(
      version_tag   = version_tag,
      activity_type = act, threshold_pct = thr,
      max_iai_trs   = miai, act_count_pct = acp,
      min_class_n   = min_class_n, reason = "< 250 per sex group"
    ))
    next
  }
  
  X <- sub %>% select(all_of(FEATURES)) %>% mutate(across(everything(), as.numeric))
  y <- factor(sub[[TARGET]], levels = c(0, 1))
  
  # ── Outer folds — guard against createFolds failure ──────────────────────
  outer_folds <- tryCatch(
    createFolds(y, k = 5, list = TRUE, returnTrain = FALSE),
    error = function(e) {
      cat("  ERROR creating outer folds:", conditionMessage(e), "\n"); NULL
    }
  )
  if (is.null(outer_folds)) next
  
  fold_metrics  <- list()
  fold_features <- list()
  
  for (fold_idx in seq_along(outer_folds)) {
    
    test_idx  <- outer_folds[[fold_idx]]
    train_idx <- setdiff(seq_len(nrow(X)), test_idx)
    
    X_train <- X[train_idx, ];  y_train <- y[train_idx]
    X_test  <- X[test_idx,  ];  y_test  <- y[test_idx]
    
    # ── BUG FIX: NZV removal BEFORE preProcess ──────────────────────────────
    nzv <- nearZeroVar(X_train, saveMetrics = FALSE)
    if (length(nzv) > 0) {
      removed_feats <- colnames(X_train)[nzv]
      cat(sprintf("  Fold %d: removing zero/near-zero variance features: %s\n",
                  fold_idx, paste(removed_feats, collapse = ", ")))
      X_train <- X_train[, -nzv, drop = FALSE]
      X_test  <- X_test[,  -nzv, drop = FALSE]
    }
    
    pp        <- preProcess(X_train, method = c("center", "scale"))
    X_train_s <- predict(pp, X_train)
    X_test_s  <- predict(pp, X_test)
    
    # ── Inner 5-fold tuning ──────────────────────────────────────────────────
    inner_folds <- tryCatch(
      createFolds(y_train, k = 5, list = TRUE, returnTrain = TRUE),
      error = function(e) NULL
    )
    if (is.null(inner_folds)) next
    
    best_acc   <- -Inf
    best_cost  <- NA
    best_gamma <- NA
    
    for (g_row in seq_len(nrow(tune_grid))) {
      c_val <- tune_grid$cost[g_row]
      g_val <- tune_grid$gamma[g_row]
      
      inner_accs <- sapply(seq_along(inner_folds), function(ifold) {
        i_train_idx <- inner_folds[[ifold]]
        i_test_idx  <- setdiff(seq_len(nrow(X_train_s)), i_train_idx)
        
        Xi_train <- X_train_s[i_train_idx, ]; yi_train <- y_train[i_train_idx]
        Xi_test  <- X_train_s[i_test_idx,  ]; yi_test  <- y_train[i_test_idx]
        
        # NZV inside inner fold
        nzv_i <- nearZeroVar(Xi_train, saveMetrics = FALSE)
        if (length(nzv_i) > 0) {
          Xi_train <- Xi_train[, -nzv_i, drop = FALSE]
          Xi_test  <- Xi_test[,  -nzv_i, drop = FALSE]
        }
        
        if (length(unique(yi_train)) < 2) return(NA)
        
        m_inner <- tryCatch(
          e1071::svm(x = Xi_train, y = yi_train, kernel = "radial",
                     cost = c_val, gamma = g_val,
                     type = "C-classification", probability = TRUE),
          error = function(e) NULL
        )
        if (is.null(m_inner)) return(NA)
        mean(predict(m_inner, Xi_test) == yi_test)
      })
      
      mean_acc <- mean(inner_accs, na.rm = TRUE)
      if (!is.na(mean_acc) && mean_acc > best_acc) {
        best_acc   <- mean_acc
        best_cost  <- c_val
        best_gamma <- g_val
      }
    }
    
    # ── Final outer model ────────────────────────────────────────────────────
    final_model <- tryCatch(
      e1071::svm(x = X_train_s, y = y_train, kernel = "radial",
                 cost = best_cost, gamma = best_gamma,
                 type = "C-classification", probability = TRUE),
      error = function(e) NULL
    )
    if (is.null(final_model)) {
      cat("  Fold", fold_idx, ": model failed, skipping.\n"); next
    }
    
    pred_obj   <- predict(final_model, X_test_s, probability = TRUE)
    pred_class <- as.integer(as.character(pred_obj))
    pred_prob  <- attr(pred_obj, "probabilities")[, "1"]
    
    fm <- compute_metrics(
      actual          = as.integer(as.character(y_test)),
      predicted_prob  = pred_prob,
      predicted_class = pred_class,
      fold_id         = fold_idx,
      best_cost       = best_cost,
      best_gamma      = best_gamma
    )
    fm$version_tag    <- version_tag
    fm$activity_type  <- act
    fm$threshold_pct  <- thr
    fm$max_iai_trs    <- miai
    fm$act_count_pct  <- acp
    fm$removed_features <- if (length(nzv) > 0) paste(removed_feats, collapse = ";") else NA
    fold_metrics[[fold_idx]] <- fm
    
    fi    <- svm_feature_importance_rbf(final_model, X_test_s, y_test, n_perm = 20)
    fi_df <- data.frame(
      version_tag   = version_tag,
      activity_type = act, threshold_pct = thr,
      max_iai_trs   = miai, act_count_pct = acp,
      outer_fold    = fold_idx,
      feature       = names(fi),
      importance    = as.numeric(fi),
      rank          = seq_along(fi),
      row.names     = NULL
    )
    fold_features[[fold_idx]] <- fi_df
    
    cat(sprintf("  Fold %d | C=%.3f γ=%.4f | Acc=%.3f AUC=%.3f\n",
                fold_idx, best_cost, best_gamma, fm$accuracy, fm$auc))
  }
  
  if (length(fold_metrics) > 0) {
    all_metrics[[version_tag]]  <- bind_rows(fold_metrics)
    all_features[[version_tag]] <- bind_rows(fold_features)
  }
}

# ── Combine & save ────────────────────────────────────────────────────────────
results_metrics  <- bind_rows(all_metrics)
results_features <- bind_rows(all_features)

summary_metrics <- results_metrics %>%
  group_by(version_tag, activity_type, threshold_pct, max_iai_trs, act_count_pct) %>%
  summarise(
    mean_accuracy     = mean(accuracy,     na.rm = TRUE),
    sd_accuracy       = sd(accuracy,       na.rm = TRUE),
    mean_sensitivity  = mean(sensitivity,  na.rm = TRUE),
    sd_sensitivity    = sd(sensitivity,    na.rm = TRUE),
    mean_specificity  = mean(specificity,  na.rm = TRUE),
    sd_specificity    = sd(specificity,    na.rm = TRUE),
    mean_ppv          = mean(ppv,          na.rm = TRUE),
    mean_npv          = mean(npv,          na.rm = TRUE),
    mean_f1           = mean(f1,           na.rm = TRUE),
    sd_f1             = sd(f1,             na.rm = TRUE),
    mean_balanced_acc = mean(balanced_acc, na.rm = TRUE),
    mean_auc          = mean(auc,          na.rm = TRUE),
    sd_auc            = sd(auc,            na.rm = TRUE),
    n_folds_completed = n(),
    .groups = "drop"
  )

top_features_summary <- results_features %>%
  group_by(version_tag, activity_type, threshold_pct,
           max_iai_trs, act_count_pct, feature) %>%
  summarise(
    mean_importance = mean(importance, na.rm = TRUE),
    mean_rank       = mean(rank,       na.rm = TRUE),
    times_top5      = sum(rank <= 5),
    .groups = "drop"
  ) %>%
  arrange(version_tag, mean_rank)

#write.csv(all_metrics,     "HEBB_SVM_Mar30_results.csv",  row.names = FALSE)
#write.csv(fold_metrics,     "HEBB_SVM_Mar30_fold_metrics.csv",   row.names = FALSE)
#write.csv(fm,    "HEBB_SVM_Mar30_fm.csv", row.names = FALSE)
#write.csv(all_features, "HEBB_SVM_Mar30_top_features.csv")
#write.csv(skipped_versions,    "svm_nested_cv_skipped_versions.csv",  row.names = FALSE)


#combined <- bind_rows(fi_df)   # or bind_rows(all_features)
#write.csv(combined, "HEBB_SVM_Mar30_fi_df.csv", row.names = FALSE)
















library(dplyr)
library(tidyr)

# ── 1. Load data ──────────────────────────────────────────────────────────────
combined <- bind_rows(all_metrics)   # or bind_rows(all_features)
df <- combined
# ── 2. Performance metrics to summarise ──────────────────────────────────────
perf_metrics <- c("accuracy", "sensitivity", "specificity",
                  "ppv", "npv", "f1", "balanced_acc", "auc")


cm_counts <- c("TP", "TN", "FP", "FN")

# ── 3. Compute summary per unique version combination ─────────────────────────
comparison_table <- df %>%
  group_by(activity_type, threshold_pct, max_iai_trs, act_count_pct) %>%
  summarise(
    # Sum confusion matrix counts across folds
    across(
      all_of(cm_counts),
      ~ sum(.x, na.rm = TRUE),
      .names = "total_{.col}"
    ),
    # Mean performance metrics across folds
    across(
      all_of(perf_metrics),
      ~ round(mean(.x, na.rm = TRUE), 4),
      .names = "mean_{.col}"
    ),
    n_folds = n(),
    .groups = "drop"
  ) %>%
  arrange(activity_type, threshold_pct, max_iai_trs, act_count_pct)

# ── 4. Preview ────────────────────────────────────────────────────────────────
print(comparison_table, n = Inf)

# ── 5. Export to CSV ──────────────────────────────────────────────────────────
#write.csv(comparison_table,
#          "HEBB_SVM_performance_comparison_table.csv",
#          row.names = FALSE)













# get best feature version to add to ICA tikhonov.
#ctivity_type linear, threshold_pct 85, max_iai_trs 200,act_count_pct 120
all_features <- read.csv('graph_features_ALL_subjects_Mar30.csv')

# ── 2. Filter to the best-performing version ───────────────────────────────────
best_version <- all_features %>%
  filter(
    activity_type  == "linear",
    threshold_pct  == 85,
    max_iai_trs    == 200,
    act_count_pct  == 120
  )

# ── 3. Select EID, activation_count, and all graph features ───────────────────
graph_features_cols <- c(
  "EID", "activation_count",
  "n_assemblies", "Nodes", "L1", "RE", "PE",
  "L2", "L3", "LCC", "LSC", "ATD",
  "Density", "Diameter", "ASP", "CC"
)

best_version_features <- best_version %>%
  select(all_of(graph_features_cols))

# ── 4. Preview ─────────────────────────────────────────────────────────────────
cat("Rows:", nrow(best_version_features), "\n")
cat("Subjects (unique EIDs):", n_distinct(best_version_features$EID), "\n")
print(head(best_version_features))

# ── 5. Save to CSV ─────────────────────────────────────────────────────────────
write.csv(best_version_features,
          "FULL_FINAL_BEST_graph_features_APR1.csv",
          row.names = FALSE)







# FILTER THE BEST FEATURES TO GET TO THE MAIN ONES WITH HIGHEST IMPORTANCE

# ── 1. Load data ───────────────────────────────────────────────────────────────
top_features <- read.csv('HEBB_SVM_Mar30_top_features.csv', stringsAsFactors = FALSE)

# ── 2. Summary table: aggregate across ALL versions and folds ─────────────────
# For each feature compute:
#   - mean_importance   : average importance score (higher = better)
#   - mean_rank         : average rank (lower = better, 1 is top)
#   - times_rank1       : how often the feature was ranked #1
#   - times_top3        : how often it appeared in the top 3
#   - pct_nonzero       : % of folds where it had non-zero importance
#   - n_appearances     : total fold × version appearances

feature_table <- top_features %>%
  group_by(feature) %>%
  summarise(
    mean_importance = round(mean(importance, na.rm = TRUE), 6),
    sd_importance   = round(sd(importance,   na.rm = TRUE), 6),
    mean_rank       = round(mean(rank,        na.rm = TRUE), 2),
    times_rank1     = sum(rank == 1),
    times_top3      = sum(rank <= 3),
    pct_nonzero     = round(100 * mean(importance != 0), 1),
    n_appearances   = n(),
    .groups = "drop"
  ) %>%
  arrange(mean_rank)   # sorted by mean rank (best features first)

# ── 3. Preview ─────────────────────────────────────────────────────────────────
print(feature_table, n = Inf)

# ── 4. Optional: version-stratified view ──────────────────────────────────────
# Mean importance per feature per version (wide format — features as rows,
# versions as columns), useful to see if dominance is consistent or version-specific
feature_by_version <- top_features %>%
  group_by(feature, version_tag) %>%
  summarise(mean_imp = round(mean(importance, na.rm = TRUE), 5),
            .groups = "drop") %>%
  pivot_wider(names_from  = version_tag,
              values_from = mean_imp,
              values_fill = NA)

# ── 5. Export both tables ──────────────────────────────────────────────────────
write.csv(feature_table,
          "best_features_overall_APR1.csv",
          row.names = FALSE)

write.csv(feature_by_version,
          "best_features_by_version_APR1.csv",
          row.names = FALSE)








#SUB BEST VERSION FEATURES TOP 12 ONLY

best_version_features_SUB <- best_version_features %>% select(-c(activation_count,n_assemblies,RE ))
names(best_version_features_SUB)

write.csv(best_version_features_SUB,
          "SUB_FINAL_BEST_graph_features_APR1.csv",
          row.names = FALSE)



