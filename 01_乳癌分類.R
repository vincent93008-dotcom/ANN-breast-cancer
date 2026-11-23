# ===========================================================================
# ANN 課堂實作（一）｜乳癌影像特徵的二元分類（Wisconsin Diagnostic）
# 研究任務：以 Logistic Regression 與單隱藏層 ANN 預測 malignant (1=惡性, 0=良性)
# 來源：UCI Breast Cancer Wisconsin (Diagnostic)；scikit-learn 內建資料副本
# 注意：本範例僅供模型教學，不能用於臨床診斷或醫療決策。
# 課堂 PPTX：01_乳癌分類_課堂練習_資料介紹.pptx
# 執行方式：在本資料夾開啟 RStudio 專案，或將工作目錄設為本資料夾後執行本檔。
# 依賴套件：nnet（R recommended package）；不需要 Python 或連線下載資料。
# ===========================================================================

# [投影片 01-04] 套件、檔案、欄位
if (!requireNamespace("nnet", quietly = TRUE)) {
  stop("尚未安裝 nnet 套件，請先執行 install.packages('nnet')。")
}
set.seed(20261025)
file_candidates <- c("breast_cancer_wdbc.csv",
                     "01_課堂練習_乳癌分類/breast_cancer_wdbc.csv",
                     "ANN_雙實例教學專案/01_課堂練習_乳癌分類/breast_cancer_wdbc.csv")
file_exists <- file_candidates[file.exists(file_candidates)]
if (length(file_exists) == 0) stop("找不到 CSV。請將工作目錄設定為案例一的資料夾，或 ZIP 解壓後的專案根目錄。")
data_file <- file_exists[1]
dat <- read.csv(data_file, stringsAsFactors = FALSE, check.names = FALSE)
selected <- c("mean_radius", "mean_texture", "mean_smoothness",
              "mean_compactness", "worst_concavity", "worst_concave_points")
stopifnot(all(c(selected, "malignant") %in% names(dat)))
stopifnot(all(dat$malignant %in% 0:1), !anyNA(dat[, c(selected, "malignant")]))

out_dir <- "Results_乳癌分類"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
cat("[STEP 1] 檢查資料\n")
cat("筆數：", nrow(dat), "；原始特徵數：", ncol(dat) - 2, "\n")
print(table(實際類別 = dat$malignant))
print(summary(dat[, selected]))
write.csv(as.data.frame(table(malignant = dat$malignant)),
          file.path(out_dir, "01_class_frequency.csv"), row.names = FALSE)

# [投影片 05-06] 依目標類別進行分層 75%/25% 訓練／測試切割
idx_0 <- which(dat$malignant == 0L)
idx_1 <- which(dat$malignant == 1L)
train_idx <- sort(c(sample(idx_0, floor(.75 * length(idx_0))),
                    sample(idx_1, floor(.75 * length(idx_1)))))
train_raw <- dat[train_idx, c(selected, "malignant"), drop = FALSE]
test_raw  <- dat[-train_idx, c(selected, "malignant"), drop = FALSE]
cat("[STEP 2] 訓練集:", nrow(train_raw), "；測試集:", nrow(test_raw), "\n")

# [投影片 07] 僅使用訓練資料估計平均值與標準差（防止資料洩漏）
train_mean <- vapply(train_raw[selected], mean, numeric(1))
train_sd <- vapply(train_raw[selected], sd, numeric(1))
train_sd[train_sd == 0] <- 1
transform_X <- function(d) {
  z <- sweep(as.matrix(d[, selected, drop = FALSE]), 2, train_mean, "-")
  z <- sweep(z, 2, train_sd, "/")
  out <- as.data.frame(z)
  out$malignant <- d$malignant
  out
}
train <- transform_X(train_raw)
test <- transform_X(test_raw)
write.csv(data.frame(variable = selected, mean_training = train_mean,
                     sd_training = train_sd),
          file.path(out_dir, "02_training_scaler.csv"), row.names = FALSE)

# [投影片 08-09] 建立 Logistic Regression（Sigmoid 單一輸出神經元）
model_logit <- glm(malignant ~ ., data = train, family = binomial())
prob_logit <- as.numeric(predict(model_logit, newdata = test, type = "response"))

# [投影片 10] 建立 ANN：6 input -> 6 hidden units -> 1 Sigmoid output
# entropy=TRUE 指定交叉熵；size=6 是隱藏層神經元數（本例固定，不在測試集調參）。
set.seed(20261025)
model_ann <- nnet::nnet(malignant ~ ., data = train,
                        size = 6, entropy = TRUE,
                        decay = 0.01, maxit = 800,
                        MaxNWts = 1000, trace = FALSE)
prob_ann <- as.numeric(predict(model_ann, newdata = test, type = "raw"))
prob_logit <- pmin(pmax(prob_logit, 1e-8), 1 - 1e-8)
prob_ann <- pmin(pmax(prob_ann, 1e-8), 1 - 1e-8)

# [投影片 11] 以惡性為陽性類別，計算分類指標
binary_metrics <- function(y, p, threshold = 0.50) {
  z <- as.integer(p >= threshold)
  TP <- sum(z == 1 & y == 1); FN <- sum(z == 0 & y == 1)
  TN <- sum(z == 0 & y == 0); FP <- sum(z == 1 & y == 0)
  safe <- function(num, den) if (den == 0) NA_real_ else num / den
  prec <- safe(TP, TP + FP)
  sens <- safe(TP, TP + FN)
  spec <- safe(TN, TN + FP)
  f1 <- if (is.na(prec) || is.na(sens) || prec + sens == 0) NA_real_ else 2 * prec * sens / (prec + sens)
  # AUC 為 Mann-Whitney 排名統計量，避免額外安裝 ROC 套件
  n_pos <- sum(y == 1); n_neg <- sum(y == 0)
  auc <- (sum(rank(p, ties.method = "average")[y == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
  c(Accuracy = mean(z == y), Sensitivity = sens, Specificity = spec,
    Precision = prec, F1 = f1, AUC = auc,
    Brier = mean((p - y)^2),
    LogLoss = -mean(y * log(p) + (1 - y) * log(1 - p)),
    TP = TP, FP = FP, TN = TN, FN = FN)
}
real <- test$malignant
report <- rbind(Logistic = binary_metrics(real, prob_logit),
                ANN = binary_metrics(real, prob_ann))
print(round(report, 4))
write.csv(data.frame(Model = rownames(report), report, row.names = NULL),
          file.path(out_dir, "03_test_metrics.csv"), row.names = FALSE)

# [投影片 12] 混淆矩陣
cm_logit <- table(Actual = factor(real, levels = 0:1),
                  Predicted = factor(as.integer(prob_logit >= .5), levels = 0:1))
cm_ann <- table(Actual = factor(real, levels = 0:1),
                Predicted = factor(as.integer(prob_ann >= .5), levels = 0:1))
cat("\nLogistic confusion matrix (0=benign, 1=malignant):\n")
print(cm_logit)
cat("\nANN confusion matrix:\n")
print(cm_ann)
write.csv(as.data.frame(cm_logit), file.path(out_dir, "04_confusion_logistic.csv"), row.names = FALSE)
write.csv(as.data.frame(cm_ann), file.path(out_dir, "05_confusion_ann.csv"), row.names = FALSE)

# [投影片 13] 改變閾值，觀察 Sensitivity/Specificity 的取捨
thresholds <- c(.30, .50, .70)
thr_tab <- do.call(rbind, lapply(thresholds, function(t) {
  a <- binary_metrics(real, prob_ann, t)
  data.frame(Threshold = t, Accuracy = a["Accuracy"],
             Sensitivity = a["Sensitivity"], Specificity = a["Specificity"],
             Precision = a["Precision"], row.names = NULL)
}))
write.csv(thr_tab, file.path(out_dir, "06_ann_thresholds.csv"), row.names = FALSE)
print(thr_tab)

# [投影片 14-15] 繪出測試集 ROC 曲線／預測機率圖
roc_points <- function(y, p) {
  t <- seq(1, 0, length.out = 201)
  out <- t(vapply(t, function(th) {
    pred <- as.integer(p >= th)
    c(FPR = sum(pred == 1 & y == 0) / sum(y == 0),
      TPR = sum(pred == 1 & y == 1) / sum(y == 1))
  }, numeric(2)))
  as.data.frame(out)
}
a <- roc_points(real, prob_logit); b <- roc_points(real, prob_ann)
png(file.path(out_dir, "07_ROC_comparison.png"), width = 1100, height = 850, res = 145)
plot(a$FPR, a$TPR, type = "l", lwd = 3, col = "#2D7185",
     xlab = "False positive rate (1 - specificity)", ylab = "True positive rate (sensitivity)",
     main = "Held-out test set: ROC comparison", xlim = c(0, 1), ylim = c(0, 1))
lines(b$FPR, b$TPR, col = "#D06F64", lwd = 3)
abline(a = 0, b = 1, lty = 2, col = "gray60")
legend("bottomright", c(sprintf("Logistic (AUC %.3f)", report[1,"AUC"]),
                        sprintf("ANN (AUC %.3f)", report[2,"AUC"])),
       lwd = 3, col = c("#2D7185", "#D06F64"), bty = "n")
invisible(dev.off())

png(file.path(out_dir, "08_predicted_probability.png"), width = 1100, height = 820, res = 145)
par(mfrow = c(1, 2))
boxplot(prob_logit ~ factor(real, levels = 0:1, labels = c("Benign", "Malignant")),
        col = c("#CFE8EB", "#F4D9D4"), ylim = c(0, 1), xlab = "Actual class",
        ylab = "P(malignant)", main = "Logistic Regression")
abline(h = .5, lty = 2)
boxplot(prob_ann ~ factor(real, levels = 0:1, labels = c("Benign", "Malignant")),
        col = c("#CFE8EB", "#F4D9D4"), ylim = c(0, 1), xlab = "Actual class",
        ylab = "P(malignant)", main = "ANN")
abline(h = .5, lty = 2)
invisible(dev.off())

# [投影片 16] 保存測試預測與模型
pred_table <- data.frame(Actual = real, Prob_Logistic = prob_logit,
                         Prob_ANN = prob_ann, Pred_Logistic = as.integer(prob_logit >= .5),
                         Pred_ANN = as.integer(prob_ann >= .5))
write.csv(pred_table, file.path(out_dir, "09_test_predictions.csv"), row.names = FALSE)
saveRDS(list(Logistic = model_logit, ANN = model_ann,
             scaler_mean = train_mean, scaler_sd = train_sd,
             features = selected, seed = 20261025),
        file.path(out_dir, "10_fitted_models.rds"))
summary_txt <- c("ANN 二元分類課堂練習", "資料：UCI Wisconsin Diagnostic Breast Cancer",
                 sprintf("原始樣本 %d；訓練 %d；測試 %d", nrow(dat), nrow(train), nrow(test)),
                 "陽性類別：惡性=1；僅以訓練資料建立標準化參數",
                 capture.output(print(round(report, 4))),
                 "Logistic confusion:", capture.output(print(cm_logit)),
                 "ANN confusion:", capture.output(print(cm_ann)),
                 "閾值敏感度分析:", capture.output(print(thr_tab)),
                 "重要：這是機器學習教材，不能用於臨床決策。")
writeLines(summary_txt, file.path(out_dir, "00_result_summary.txt"), useBytes = TRUE)
cat("\n完成。分析表格、兩張 PNG 與模型 RDS 已存於：", normalizePath(out_dir), "\n")
