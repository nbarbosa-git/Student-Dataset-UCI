
Pre_load <- function(){

library("pdist")
library("ggplot2")
library("gridExtra")

#Read CSV
Raw_Dataset=read.table("student-por.csv",sep=";",header=TRUE)

#Features Transformation (factor to binary)
Raw_Dataset <- Raw_Dataset[,c(-11)] #delete reason column
for (j in 1:ncol(Raw_Dataset)){
  
  
  
  if(is.factor(Raw_Dataset[,j])){
    Raw_Dataset[,j] <- as.numeric(Raw_Dataset[,j])  
    for (i in 1:nrow(Raw_Dataset)){
      
      if(j==12){ #Guardian Column
        if( Raw_Dataset[i,j]==3){Raw_Dataset[i,j]=0} else {Raw_Dataset[i,j]=1} 
      }
      
      else if( Raw_Dataset[i,j]==1){Raw_Dataset[i,j]=0} else {Raw_Dataset[i,j]=1}
    } }
  Raw_Dataset[,j] <- as.numeric(Raw_Dataset[,j]) 
}

#Balancing Dataset
for (i in 1:nrow(Raw_Dataset)){
  if( (Raw_Dataset[i,7]==0)|(Raw_Dataset[i,7]==1)|(Raw_Dataset[i,7]==2)) {Raw_Dataset[i,7]=0} #Compile Medu
  if( (Raw_Dataset[i,7]==3)|(Raw_Dataset[i,7]==4)) {Raw_Dataset[i,7]=1} #Compile Medu
  if( (Raw_Dataset[i,8]==0)|(Raw_Dataset[i,8]==1)|(Raw_Dataset[i,8]==2)) {Raw_Dataset[i,8]=0} #Compile Fedu
  if( (Raw_Dataset[i,8]==3)|(Raw_Dataset[i,8]==4)) {Raw_Dataset[i,8]=1} #Compile Fedu
}


#Outliers Dettection
scaled.Raw_Dataset <- scale(Raw_Dataset)  
#boxplot(scaled.Raw_Dataset,las = 2)
mean_distance = vector(length = nrow(Raw_Dataset))
for (i in 1:nrow(Raw_Dataset)){
  euclidian_dist = pdist(scaled.Raw_Dataset, indices.A = i, indices.B = c(-i))
  mean_distance[i]=  mean(as.matrix(euclidian_dist))
  outliers_index=which(mean_distance > 10) #10 was chosen after an visual analysis of the mean values
}

scaled.Raw_Dataset <- scaled.Raw_Dataset[c(-outliers_index),]
Raw_Dataset <- Raw_Dataset[c(-outliers_index),]



piecharts <- function(value, pie_label){
  df <- data.frame(
    variable = c("Até Educação Primária (4o Ano)"," 5o ao 9o ano","Escola Secundária","Ensino Superior"),
    value = c(value))
  
  graf <- ggplot(transform(transform(df, value=value/sum(value)), labPos=cumsum(value)-value/2), 
                 aes(x="", y = value, fill = variable)) +
    geom_bar(width = 1, stat = "identity") +
    scale_fill_manual(values = c("red", "yellow","blue", "green", "cyan")) +
    coord_polar(theta = "y") +
    labs(title = pie_label) + 
    geom_text(aes(x=1.2, y=labPos, label=scales::percent(value)))
  return(graf)
}


#Pie charts
Mother_Edu_Ratio <- summary(as.factor(Raw_Dataset$Medu))
Father_Edu_Ratio <- summary(as.factor(Raw_Dataset$Fedu))
#graf_mom <- piecharts(Mother_Edu_Ratio, "Mother Education")
#graf_dad <- piecharts(Father_Edu_Ratio, "Father Education")
#grid.arrange(graf_mom , graf_dad , ncol=2)



  return(scaled.Raw_Dataset)
}