
# Vietnamese Market Indices Prediction Project
This project focuses on forecasting Vietnam’s major stock indices—VN-Index and VN30—using a suite of deep learning models, including LSTM, Transformer, and NeuralProphet. 
It combines historical market data with external financial indicators to train and evaluate both multi-step and multi-output forecasting architectures. 
The goal is to capture short-term market trends and provide accurate multi-day predictions that can support financial analysis and decision-making.

Below is a breakdown of what each component of the codebase does.

## 1. Data Collection and Preprocessing
The central dataset for this study is the VN-Index, which reflects the aggregated performance of all stocks listed on the Ho Chi Minh Stock Exchange (HOSE). A multi-stage pipeline was developed to scrape, clean, and prepare this data using custom Python scripts.

* VN-Index Data Collection
Raw VN-Index data was collected from the `StockBiz.vn` website using Selenium WebDriver, as implemented in the `vn_index_scripts/scrape_vn_index.py` script. This automated browser-based scraping script:

    - Opens the VN-Index stats webpage.

    - Waits for the calendar widget and data table to load.

    - Collects daily index statistics including date, VN-Index close, volume, value, and price change.

    - Saves the output in CSV format.

This approach allows access to granular historical data directly from a Vietnamese financial source that is otherwise not available via standard APIs.

* VN-Index Cleaning and Formatting
The scraped dataset (`vn_index_data/hose_historical_data.csv`) is preprocessed in `vn_index_scripts/vn_index_preprocessing.py`, which performs:

    - Deduplication: All duplicate rows are dropped to avoid over-representation of dates.

    - Column Filtering: Only the essential columns — Date, VN-INDEX, Total Volume, and Total Value — are retained. The 'Change' column is excluded at this stage to be handled separately.

    - Null Handling: Rows with missing values are removed.

    - Date Parsing: The 'Date' column is converted from string format to datetime*using the '%m/%d/%Y' format to ensure temporal consistency.

The cleaned output is stored as `ready_data/cleaned_vn_index_data.csv`.

* External Market Data Collection
To enrich the forecasting model, a selection of international economic indicators was collected using the yahooquery API in `external_scripts/external_data.py`. This includes historical daily data for:

    - U.S. stock indices: S&P *(^GSPC), Dow Jones (^DJI), NASDAQ (^IXIC)

    - Asian indices: Shanghai Composite (*SS), Hang Seng Index (^HSI), KOSPI (^KS11)

    - Commodities: Crude Oil (CL=F), Gold (GC=F)

    - Currency pairs: USD/VND exchange rate (USDVND=X)

    - Global ETFs: Emerging Markets ETF (EEM), and Vietnam-specific ETF (VNM)

Data spans from July 28th, 2000 to the present. These files are stored individually in the external_data/ folder.

* External Data Preprocessing
The `external_scripts/external_preprocessing.py` script processes all CSVs in the external_data directory. For each file:

    - The first two header rows are skipped to correct formatting issues.

    - Columns are renamed to standardized names (Date, Close).

    - Date columns are parsed using pd.to_datetime() with format inference ('mixed') and localized to UTC, then converted to timezone-naive datetime.date.

This results in consistently formatted datasets ready for merging and alignment.

* Merging VN-Index and External Data
Using `vn_index_scripts/merge_vn_index.py`, the cleaned VN-Index data is merged with the external indicators on the Date column. Before merging:

    - A utility function remove_all_zeros() is called to drop any columns in external datasets that are entirely zero (likely due to missing or failed API pulls).

    - The merge is performed as a left join with VN-Index as the anchor to preserve its trading calendar.

This produces a master file `ready_data/vn_index_merged_data.csv`, which integrates both domestic and global economic signals.

* Capturing Price Change Dynamics
To analyze the VN-Index’s daily percentage or absolute changes, a separate script `change_scripts/change_preprocessing.py` extracts and cleans the Change column from the raw scrape. It converts the date format and aligns it with other datasets.

The processed `ready_data/change.csv` is then joined with other data sources in `change_scripts/merge_change.py`, which generates:

    - `ready_data/change_vn_index.csv` – VN-Index with corresponding price changes

    - `ready_data/change_external.csv` – External features with price changes

    - `ready_data/change_merged.csv` – Full combined dataset for modeling volatility or directional movement

## 2. Feature Engineering and Data Preparation
To enhance the model's ability to capture market dynamics, a robust set of engineered features was created. These are implemented in `general_scripts/features_engineering.py` and include classic and custom technical indicators:

    - Relative Strength Index (RSI): Computed with a rolling window, captures momentum by quantifying gain vs. loss magnitude.

    - STL Decomposition: Seasonal-Trend decomposition using LOESS is applied for isolating trend and seasonality components in price signals.

    - Rolling Statistics: Moving averages and rolling standard deviations help the model detect short-term trends and volatility bursts.

These features are added directly to the core time series DataFrame prior to scaling and sequence construction.

In `general_scripts/helper.py`, utilities for data splitting, reproducibility (set_seed), and customized loss computation are provided. Notably, a custom DirectionalLoss class is defined, penalizing the model more for predicting the wrong direction (up vs. down), reflecting practical trading concerns.

## 3. Model Architectures
This study employs two customized architectures: one designed for multi-step forecasting and the other for multi-output forecasting, both tailored for predicting the VN-Index over multiple future trading days.

* Multi-Step LSTM (Seq*eq with teacher forcing)
Implemented as `LSTMModelMultiStep`, this model uses a sequence-to-sequence structure with separate encoder and decoder LSTM layers. It predicts future values step-by-step, where each forecasted value is used to predict the next. Key features include:

    - Encoder: Processes the historical input sequence and returns a final hidden state.

    - Decoder: Initialized with the encoder state and a single timestep (e.g., the last known close price), it recursively generates future values one at a time.

    - Teacher Forcing: During training, ground truth values can be injected into the decoder at each step with a certain probability, improving gradient flow and convergence stability.

    - Flexible Horizon: The number of prediction steps (n_forecasts) is configurable, making this approach suitable for varying forecast lengths.

This autoregressive approach models the temporal dependency between each predicted timestep, allowing it to adaptively refine its outlook as each future point is generated.

* Multi-Output LSTM (Direct mapping)
Defined in `LSTMModelMultiOutput`, this model produces all future values in a single forward pass, making it a fully parallel, multi-output forecaster. Unlike the multi-step model, it does not rely on sequential decoding.

Single LSTM Encoder: Processes the input time series and returns the final hidden state.

Fully Connected Layer: Maps the hidden state to a flat vector representing all n_forecasts future values.

Output Reshaping: The flattened forecast vector is reshaped to match the desired output shape (batch_size, n_forecasts, 1).

This architecture excels at modeling the joint distribution of future values simultaneously, which can be more computationally efficient and less prone to error accumulation compared to step-by-step methods.

* Multi-Step Transformer (Seq*eq with learned queries)
The `TimeSeriesTransformerMultiStep` class implements a sequence-to-sequence Transformer model, where a learned decoder is used to produce multiple future predictions sequentially, guided by contextual memory from the encoder.

Key components:

    - Input Projection & Positional Encoding: The input sequence is first passed through a linear projection layer and augmented with sinusoidal positional encodings to preserve temporal order.

    - Encoder: A stack of Transformer encoder layers processes the entire input sequence and produces a memory representation.

    - Learned Query Embeddings: The decoder is driven by a fixed set of learnable vectors (one for each forecast step) that serve as abstract representations of each future timepoint.

    - Decoder: These query embeddings attend to the encoder memory to generate predictions for each forecast horizon step.

    - Output Head: A final linear layer maps the decoder output to the predicted VN-Index values.

This architecture supports multi-step forecasting, producing the full sequence of future predictions in a single forward pass while still modeling attention-based dependencies for each individual step.

* Multi-Output Transformer (Encoder-only with global pooling)
The `TimeSeriesTransformerMultiOutput` model simplifies the architecture by using only an encoder, without a decoder or learned queries. It is designed for direct multi-output prediction, where the entire future window is predicted from the compressed representation of the input sequence.

Key characteristics:

    - Encoder-Only Architecture: The model uses stacked Transformer encoder layers to process the input sequence and extract temporal patterns.

    - Global Average Pooling: After encoding, a global average pooling layer condenses the sequence into a single fixed-length feature vector.

    - Fully Connected Output Head: This vector is passed through a feedforward neural network that directly produces all n_forecasts predictions simultaneously.

    - Output Reshaping: The predictions are reshaped into the final output format (batch_size, forecast_horizon, 1).

This direct multi-output formulation is computationally efficient and ideal for scenarios where all future predictions are generated at once and no autoregressive reasoning is needed.

* Optional: Temporal Attention Extension
In addition to the core LSTM and Transformer models, your architecture includes an optional temporal attention mechanism (TemporalAttention in `general_scripts/attention_lstm.py`). This layer allows the decoder to dynamically attend to the most relevant parts of the encoded input sequence for each prediction step, improving interpretability and potentially accuracy.

It works by:

Computing similarity scores between each decoder query and encoder outputs.

Generating a context vector as a weighted sum of encoder states.

Injecting the context into the decoder for each forecast step.

This mechanism enhances both the multi-output and multi-step variants by allowing the model to focus selectively on past events that are most predictive of future changes.

## 4. Training and Evaluation
* The training process is handled in `general_scripts/training_evaluation.py`. Key components include:

    - Loss Functions: Both MSE and the custom DirectionalLoss can be used independently or combined.

    - Optimizer: The Adam optimizer with optional learning rate scheduling is used for training stability.

    - Batching: Time-series aware batching via TensorDataset and DataLoader.

    - Logging: Built-in logging captures training metrics and outputs for tracking model progress.

* Evaluation functions report on:

    - RMSE, MAE, R² for numeric accuracy

    - Directional Accuracy to assess movement prediction quality

Results are logged and optionally visualized.

## 5. Hyperparameter Tuning
* Using Optuna, a robust hyperparameter search framework, model optimization is automated in `general_scripts/pipelines.py`:

    - The objective() function defines the parameter search space (e.g., hidden size, dropout, learning rate).

    - Trial results are evaluated using validation RMSE or custom metrics.

    - The best parameters are selected and passed into final training runs.

* This tuning framework accelerates model development and avoids manual grid search.

## 6. Model Pipeline and Forecasting
* The full pipeline is orchestrated in `general_scripts/pipelines.py`. It supports:

    - Data normalization and sequence generation using create_sequences()

    - Model selection (LSTM vs. Transformer)

    - Training and validation loops

    - Saving best-performing model weights

    - Evaluation on test data

* Finally, in `general_scripts/predict.py`, the future_change_prediction() function allows users to:

    - Generate forecasts for the next n trading days

    - Plot actual vs. predicted curves

    - Optionally add cyclical embeddings (e.g., weekday, month) to capture market seasonality

    - The prediction function uses the trained model and standardized data to simulate live future forecasts.
