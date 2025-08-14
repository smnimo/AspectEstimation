library(corrplot)

#D <- read.csv("./giventask.csv", header=TRUE)
#D <- read.csv("./Akq_all_topic3.csv", header=TRUE)
#D <- read.csv("./Akq_spe_topic3.csv", header=TRUE)
D <- read.csv("./Akq_spe_topic4.csv", header=TRUE)

C <- ncol(D)
R <- nrow(D)
Cs <- C/2+1.5
N <- as.matrix(D[Cs : C])
dimnames(N) <- list(D[1 : R, 1], Cs : C)
col3 <- colorRampPalette(c("red", "white", "blue"))
corrplot(N, method = "circle", cl.lim = c(min(N),max(N)), col=col3(200))
Cs
