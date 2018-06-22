library("fBasics")
library("grid")
library("gridExtra")
library("corrplot")
library("Amelia")
library("pdist")

#Read CSV
Dataset=read.table("student-por.csv",sep=";",header=TRUE)

#Features Transformation (factor to binary)
Dataset <- Dataset[,c(-11)] #delete reason column
for (j in 1:ncol(Dataset)){

  
  
  if(is.factor(Dataset[,j])){
  Dataset[,j] <- as.numeric(Dataset[,j])  
  for (i in 1:nrow(Dataset)){
    
    if(j==12){ #Guardian Column
      if( Dataset[i,j]==3){Dataset[i,j]=0} else {Dataset[i,j]=1} 
    }
    
    else if( Dataset[i,j]==1){Dataset[i,j]=0} else {Dataset[i,j]=1}
  } }
  Dataset[,j] <- as.numeric(Dataset[,j]) 
  }




#Basic Stats Table
Stat_tab <- basicStats(Dataset)
Stat_tab <- t(Stat_tab[c(1,3:8,14),])
#grid.table(Stat_tab)


#tabela com valores ausentes
  #  missingness map
  missmap(Dataset, col = c("red", "navyblue"), y.cex = 0.5, x.cex = 0.8)

  # Count NA in Dataframe
  count_na <- function(x){
    sum(is.na(x))
  }
  #  % if NAs in dataframe
  percent_na <- function(x){
    mean(is.na(x))
  }
  # Proportion of NAs by column
  na_by_col <- function(df){
    sapply(df, percent_na)
  }
  # proportion of NAs by row
  na_by_row <- function(df){
    apply(df, 1, percent_na)
  }


  
#Outliers Dettection
scaled.dataset <- scale(Dataset)  
boxplot(scaled.dataset,las = 2)

mean_distance = vector(length = nrow(Dataset))
for (i in 1:nrow(Dataset)){
euclidian_dist = pdist(scaled.dataset, indices.A = i, indices.B = c(-i))
mean_distance[i]=  mean(as.matrix(euclidian_dist))
outliers_index=which(mean_distance > 10) #10 was chosen after an visual analysis of the mean values
}
plot(sort(mean_distance))
scaled.dataset <- scaled.dataset[c(-outliers_index),]
Dataset <- Dataset[c(-outliers_index),]

#Corr Matrix
COR <- cor(scaled.dataset)
corrplot(COR, method="circle")


#histograms
par(mfrow=c(4,4))
hist(Dataset$age)
hist(Dataset$Medu)
hist(Dataset$Fedu)
hist(Dataset$traveltime) 
hist(Dataset$studytime)
hist(Dataset$failures)
hist(Dataset$famrel) 
hist(Dataset$freetime)
hist(Dataset$goout)
hist(Dataset$Dalc)
hist(Dataset$Walc)
hist(Dataset$health)
hist(Dataset$absences)
hist(Dataset$G1)
hist(Dataset$G2)
hist(Dataset$G3)


library(psych)

# seleciona quatro variáveis (para deixar a figura clara)
df <- mtcars[ , c("mpg", "disp", "hp", "wt")]

df <- Load.Dataset[ , c("age", "Medu", "Fedu", "traveltime","studytime","failures","famrel","freetime",
                        "goout","Dalc","Walc","health","absences","G1","G2","G3")]


df <- scaled.Dataset
# gera a figura
pairs.panels(df)


