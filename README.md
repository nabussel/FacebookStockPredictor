# Facebook Stock Price Prediction with LSTMs  

This repository contains a deep learning project that applies **Long Short-Term Memory (LSTM) networks** to forecast **Facebook stock opening prices** using historical time series data. It demonstrates an end-to-end workflow in **Python** with widely used data science and machine learning libraries.  

---

## Technologies Used  
- **Python** – core programming language  
- **Pandas & NumPy** – data handling and preprocessing  
- **Scikit-learn** – MinMax scaling for normalization  
- **TensorFlow / Keras** – building and training deep learning models  
- **Matplotlib** – visualization of results  

---

## Project Workflow  
1. **Data Preprocessing**  
   - Load training (`fb_train.csv`) and testing (`fb_test.csv`) datasets  
   - Extract the **Open** price column  
   - Normalize values between 0–1 with `MinMaxScaler`  
   - Generate 60-day sliding windows to predict the 61st day  

2. **Model Architecture**  
   - Stacked **LSTM layers** (4 layers, 100 units each)  
   - **ReLU activations** to introduce non-linearity  
   - **Dropout layers** (20%) for regularization  
   - Final **dense layer** for regression output  

3. **Training**  
   - Optimizer: **Adam**  
   - Loss function: **Mean Squared Error (MSE)**  
   - 100 epochs, batch size of 32  

4. **Testing & Prediction**  
   - Combine training and testing sets to preserve temporal continuity  
   - Apply the same scaling and reshaping steps  
   - Predict stock prices and **inverse transform** them back to original scale  

5. **Visualization**  
   - Plot **actual vs. predicted Facebook stock prices** for comparison  

---

## Results & Applications  
- Produces next-day forecasts of Facebook stock opening prices  
- Demonstrates how to preprocess and model sequential financial data  
- Serves as a template for applying LSTMs to other time series problems  
  

