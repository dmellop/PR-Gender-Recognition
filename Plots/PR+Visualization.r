
# Read in the data

res <- read.csv("results.csv")
head(res)

library(ggplot2)


## Plot of actual and found count

x = c(1:nrow(res))

p11 <- ggplot(data = res) + 
geom_line(aes(x=x, y=Actual.Count, col = "Actual Count"), size = 0.7)+
geom_line(aes(x=x, y=Results, col = "Found by Algorithm"), size = 0.7)+
theme_bw()+
ggtitle("Actual Count vs Found by Algorithm")+
xlab("Number of the Video")+
ylab("Count of Persons")
p11

ggsave("p11")


# Get RMSE of results

categories <- unique(res$Category)
categories <- lapply(categories, as.character)
cat_errors <- matrix(NA, nrow = length(categories), ncol = 2)

for(i in c(1:length(categories)))
{
    temp <- res[res$Category == categories[i],]
    rmse <- sqrt(mean((temp$Actual.Count - temp$Results)^2))
    cat <- categories[[i]]
    cat_errors[i,] <- cbind(cat, as.numeric(rmse))
}

colnames(cat_errors) <- c("Category", "RMSE")
cat_errors[1,1] <- "Apple"
cat_errors


mean(as.numeric(cat_errors[,2]))


# Plot RMSE-s
p <- ggplot()+
geom_bar(aes(x = cat_errors[,1], y = as.numeric(cat_errors[,2])), stat='identity')+
theme_bw()+
ggtitle("RMSE of Categories")+
ylab("Root Mean Square Error")+
xlab("Category")+
theme(axis.text.x = element_text(angle = 90, hjust = 1))+
geom_text(aes(x=cat_errors[,1], y = as.numeric(cat_errors[,2]), label=cat_errors[,1], hjust = -0.05, angle = 90),
          col = "black")+
theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())+
ylim(c(0,20))

p


ggsave(filename="count_RMSE.png", plot=p, height = 2, width = 6)


# Accuracy of finding gender majority.
head(res)

res$Majority.Females <- res$Females > res$Males  # Majority of people in videos are female
res$Majority.Found.Females <- res$Found.Females > res$Found.Males  # Majority of people in videos are female
accuracy <- sum(res$Majority.Females == res$Majority.Found.Females)/nrow(res)*100
paste("Overall accuracy:", accuracy)
accuracy

categories <- unique(res$Category)
categories <- lapply(categories, as.character)
cat_acc <- matrix(NA, nrow = length(categories), ncol = 2)

for(i in c(1:length(categories)))
{
    temp <- res[res$Category == categories[i],]
    females <- temp$Females > temp$Males  # Majority of people in videos are female
    found.females <- temp$Found.Females > temp$Found.Males  # Majority of people in videos are female
    accuracy <- sum(females == found.females)/nrow(temp)*100
    cat <- categories[[i]]
    cat_acc[i,] <- cbind(cat, as.numeric(accuracy))
}

colnames(cat_acc) <- c("Category", "Accuracy")
cat_acc[1,1] <- "Apple"
cat_acc

p2 <- ggplot()+
geom_bar(aes(x = cat_acc[,1], y = as.numeric(cat_acc[,2])), stat='identity')+
geom_text(aes(x=cat_acc[,1], y = as.numeric(cat_acc[,2]), label=cat_acc[,1], hjust = 1.1, angle = 90),
          col = "white")+
theme_bw()+
ggtitle("Gender Recognition Accuracy of Categories")+
ylab("Accuracy")+
xlab("Category")+
ylim(0, 100)+
theme(axis.text.x = element_text(angle = 90, hjust = 1))+
theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

p2

#ggsave("gender_accuracy.png", p2)
ggsave(filename="gender_accuracy.png", plot=p2, height = 2.1, width = 6)
