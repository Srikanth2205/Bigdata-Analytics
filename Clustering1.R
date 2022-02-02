library(dplyr)
library(ggplot2)
library(ggpubr)
library(factoextra)
library(reshape2)
library(tidyr)

Seer_data=read.csv("./A1_BC_SEER_data.csv")
#summary(Seer_data)

Sd_r <- Seer_data %>% select(SEER.registry,Marital.status.at.diagnosis,Race.ethnicity,Sex,Primary.Site,Laterality, Histology.recode...broad.groupings,ER.Status.Recode.Breast.Cancer..1990..,Breast...Adjusted.AJCC.6th.Stage..1988.2015.,PR.Status.Recode.Breast.Cancer..1990..,surg.combine)
summary(Sd_r)

heatMap<-function(data){
  correlation = round(cor(data),2)
  melted_cor <- melt(correlation)
  
  ggplot(melted_cor, aes(x=Var1, y=Var2, fill=value)) + 
    geom_tile()
}


refined_data = select(Sd_r,-ER.Status.Recode.Breast.Cancer..1990..)
heatMap(refined_data)



#pairs(refined_data, pch = 19, lower.panel = NULL, )
#correlation Suggests that Pr and Er status has high correlation

#Outliers Detection
outliers <- boxplot(Sd_r, plot=TRUE)$out

set.seed(12345)
#kmeans.ani(Sd_r, 15)



kmean_withinss <- function(k) {
  cluster <- kmeans(Sd_r, k)
  return (cluster$tot.withinss)
}

# Setting the maximum cluster value
maxClusters <-25
# Running the algorithm over a range of k values set above
wss <- sapply(2:maxClusters, kmean_withinss)

elbow <-data.frame(2:maxClusters, wss)

ggplot(elbow, aes(x = X2.maxClusters, y = wss)) +
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = seq(1, 20, by = 1))

#k value from the elbow is selected
K_result <-kmeans(Sd_r, 8,nstart =25)
#library(cluster)
fviz_cluster(K_result, Sd_r,
             #
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)

K_result$cluster
K_result$centers

#Number of elements in each cluster
K_result$size	
