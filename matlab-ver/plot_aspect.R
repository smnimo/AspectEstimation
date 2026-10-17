library(corrplot)

# aspects p.vz (15 x K, rows = attribute values, columns = aspects) as written by learn_aspects_from_mask.m /
# calc_matching.m (csvwrite, no header); paper Fig. 9
# The values are plotted as they are, as in the paper. (The CSV files used in 2019 were edited by hand: header,
# row-index column and a copy multiplied by 2 for visibility, which this script plotted; the paper used unscaled values.)
# Newer corrplot versions call cl.lim "col.lim".
file <- "./results/Akq_spe_topic3.csv"   # paper Fig. 9: Akq_all_topic3.csv (b), Akq_spe_topic3.csv (c), Akq_spe_topic4.csv (d)

N <- as.matrix(read.csv(file, header = FALSE))
dimnames(N) <- list(1 : nrow(N), 1 : ncol(N))
col3 <- colorRampPalette(c("red", "white", "blue"))
corrplot(N, method = "circle", cl.lim = c(min(N), max(N)), col = col3(200))
