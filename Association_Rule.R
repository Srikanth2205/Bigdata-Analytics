#install.packages("arules")
#install.packages("arulesViz")

library(arules)
library(arulesViz)

# Read the data file
data<-read.csv('A1_success_data.csv')
summary(data)

# Generate frequent itemsets
items <- apriori(data,parameter = list(supp=0.1, conf=0.5,minlen=3,maxlen=4,target="frequent itemset"))
summary(items)
inspect(items)

# Creating association rules
rule_model <- apriori(data,parameter = list(supp=0.1, conf=0.5,minlen=3,maxlen=4,target="rules"))
summary(rule_model)
inspect(rule_model)

# Setting rhs as sole attribute "Success"
rhs_success<-subset(rule_model, (rhs %in% paste0("Success=", unique(data$Success))))
inspect(rhs_success)

# relationship among support, confidence, and lift
plot(rhs_success@quality)

# graph visualization based on the sorted lift value
graph_vis_lift<- head(sort(rhs_success, by='lift'),10)
inspect(graph_vis_lift)
plot(graph_vis_lift, method='graph',engine="htmlwidget")

