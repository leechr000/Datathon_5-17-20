library(dplyr)

df <- read.csv("C:\\Users\\eelrs\\Documents\\Datathon_Spring\\grocery_store_data_cleaned.csv",
                stringsAsFactors = FALSE)
#df <- df %>% filter(CATEGORY == 'Citrus')

dt <- df %>% pull(DATE)

df$DATE <- as.Date(strptime(dt, "%m/%d/%Y"))

df2 <- df %>% group_by(DATE, NAME) %>% 
  summarise('UNIT_PRICESELL' = mean(UNIT_PRICESELL),
            'UNIT_PRICEBUY' = mean(UNIT_PRICEBUY),
            'TOTAL_PROFIT' = sum(PROFIT))





write.csv(df2, file="C:\\Users\\eelrs\\Documents\\Datathon_Spring\\bydate.csv")
