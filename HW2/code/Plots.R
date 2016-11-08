# Saves relevant figures
#install.packages("ggplot2")
require("ggplot2")

location = "/Users/aciupan/ml_6867/HW2/code/data/"
#Problem 2
datP2 = read.csv("/Users/aciupan/ml_6867/HW2/code/data/PB2Data.csv", header = TRUE)
i<-1
rest_dat = datP2[datP2$Dataset==i,]

for (i in 1:4){
  # Margin
  q_1 <- ggplot(aes(x=C, y=Margin, color=factor(Model)), data = rest_dat) + 
    geom_point() + geom_line() + scale_x_log10() + theme_bw() +
    theme(legend.position="none")
  
  # Number of support vectors
  q_2 <- ggplot(aes(x=C, y=NumSupportVectors, color=factor(Model)), data = rest_dat) + 
    geom_point() + geom_line() + scale_x_log10() + theme_bw()
  
  # Number of margin support vectors
  q_3 <- ggplot(aes(x=C, y=NumSupportVectorsAtMargin, color=factor(Model)), data = rest_dat) + 
    geom_point() + geom_line() + scale_x_log10() + theme_bw() 
  
  # Save
  ggsave(filename = paste0(location,"graphs/D",i,"Margin.png"),
         plot=q_1)
  ggsave(filename = paste0(location,"graphs/D",i,"NumSV.png"),
         plot=q_2)
  ggsave(filename = paste0(location,"graphs/D",i,"NumSVMargin.png"),
         plot=q_3)
}

#Problem 3
datP3p2 = read.csv("/Users/aciupan/ml_6867/HW2/code/data/PB3p2Data.csv", header = TRUE)
datP3p2$X <- NULL
q_p3p2 <- ggplot(aes(x= Lambda, y=Margin), data = datP3p2) + geom_line()+
  theme_bw() + scale_x_log10()
ggsave(filename = paste0(location,"graphs/","MarginVsLambda.png"),
       plot=q_p3p2)

#Problem 1
dat = read.csv("/Users/aciupan/ml_6867/HW2/code/data/data1_test.csv",
               sep = " ",
               header = FALSE)
colnames(dat) = c("X1", "X2", "Y")
fit <- glm(Y~X1+X2,
           data=dat,
           family=binomial())

