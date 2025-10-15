 #!/bin/bash
# Base command
BASE_CMD="python -m training_pipeline.train --tasks churn propensity_category propensity_sku --accelerator gpu --devices 2 --disable-relevant-clients-check --clearml-project unsupervised_metrics/gru_base_embeddings_metrics"

# Loop through each embedding configuration
for embedding in $DATA_DIR/unsupervised_gru_base/*/; do
    # Set the embeddings directory and log name
    LOG_NAME="$(basename "$embedding")"
            
    EMBEDDINGS_DIR="${DATA_DIR}/unsupervised_gru_base/${LOG_NAME}"
    
    # Construct the full command
    FULL_CMD="${BASE_CMD} --embeddings-dir ${EMBEDDINGS_DIR} --log-name ${LOG_NAME} --data-dir ${DATA_DIR}/raw "
    
    # Print the command for verification
    echo "Running: ${FULL_CMD}"
    
    # Execute the command
    eval "${FULL_CMD}"
    
    # Check if the command succeeded
    if [ $? -ne 0 ]; then
        echo "Error encountered with ${LOG_NAME}"
    fi
done
