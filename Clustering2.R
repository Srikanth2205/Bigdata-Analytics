cancer_data <- read.csv("A1_BC_SEER_data.csv")
features <- c("SEER.registry", "Marital.status.at.diagnosis", "Race.ethnicity", "Sex", 
              "Primary.Site", "Laterality", "Histology.recode...broad.groupings", 
              "ER.Status.Recode.Breast.Cancer..1990..", "PR.Status.Recode.Breast.Cancer..1990..", 
              "Breast...Adjusted.AJCC.6th.Stage..1988.2015.", "surg.combine")
A1_data = cancer_data[,(names(cancer_data) %in% features)]

summary(A1_data)

hist(A1_data$Marital.status.at.diagnosis,probability=T, main="Histogram of normal data",
     xlab="Approximately normally distributed data")

#visualization
install.packages("ggfortify")
library(ggfortify)
pca_A1 <- prcomp(A1_data, center = TRUE, scale. = TRUE)
summary(pca_A1)

autoplot(pca_A1)
plot(pca_A1, type="lines")

set.seed(123)

#best value of dbscan after trial and error
library(dbscan)
gc()

dbscan_model_1 <- dbscan(pca_A1$x[,1:2], eps = 0.05, minPts = 3)
dbscan_model_2 <- dbscan(pca_A1$x[,1:2], eps = 0.05, minPts = 500)
dbscan_model_3 <- dbscan(pca_A1$x[,1:3], eps = 0.1, minPts = 750) 
dbscan_model_4 <- dbscan(pca_A1$x[,1:4], eps = 0.2, minPts = 1500)# best model 30% noise points approx
dbscan_model_5 <- dbscan(pca_A1$x[,1:9], eps = 0.6, minPts = 1500)

hullplot(pca_A1$x, dbscan_model_4, main = "DBScan")
