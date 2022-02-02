library(ggplot2)
task1_data <- read.csv('A1_performance_test.csv')
summary(task1_data)
# q plot stands for quick plot which is used to plot the data
qplot(x = task1_data$approach,
      y = task1_data$performance)
noapproach<-subset(task1_data,approach=='no_approach')
#get the performance of no_approach
x<-noapproach[,2]

approach12<-subset(task1_data,approach=='approach1'|approach=='approach2')
#get the performance of approach1 and approach2
y<-approach12[,2] 

#Hypothesis test for on no_approach Vs approach1 and approach2
wilcox.test(x,y,alternative = 'less',mu=mean(x),sigma.x = sd(x),sigma.y = sd(y), conf.level = 0.95,conf.int = 'TRUE')

noapproach<-subset(task1_data,approach=='approach1')
#get the performance of approach1
x<-noapproach[,2]
approach12<-subset(task1_data,approach=='approach2')
#get the performance of approached
y<-approach12[,2]

#Hypothesis test for on approach1 Vs approach2
wilcox.test(x,y,alternative = 'less',mu=mean(x),sigma.x = sd(x),sigma.y = sd(y), conf.level = 0.95,conf.int = 'TRUE')

