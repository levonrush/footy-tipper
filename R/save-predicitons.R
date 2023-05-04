save_predictions <- function(predictions, inference_df){
  
  if (prod_run == TRUE){
  
    write.csv(x = predictions, file = "predictions.csv")
    
    # Upload the file
    drive_upload(media = "predictions.csv",
                 path = "footy-tipping-predictions/",
                 name = paste0("round", unique(inference_df$round_id), "_",
                               unique(inference_df$competition_year), ".csv"), 
                 type = NULL,
                 overwrite = TRUE)
    
    unlink("predictions.csv")
  
  }
  
}
