# Deep Learning Framework for HAR
This repository provides a comparison of deep learning models for Human Activity Recognition (HAR) to support independent living in smart homes. We compare the HAR performance of a Tiny Transformer Encoder (tiny_transformer), Transformer Encoder (transformer_encoder), Convolutional Neural Network (cnn), Gated Recurrent Unit (gru) and Deep Convolutional LSTM (deep_conv_lstm) for fitting the Inertial Measurement Unit (IMU) data found here: https://doi.org/10.25919/d7xf-n080.   

### Previous studies that used the same dataset can be found here:    
1- Moid Sandhu, et al. "Feasibility of motion sensor-based human activity recognition for supporting independence in smart homes." Maturitas 199 (2025): 108632. https://www.sciencedirect.com/science/article/pii/S0378512225004402    

2- Moid Sandhu, et al. "Fusing IoT Wearable and Object Motion Sensors for Enhanced Activity Recognition in Smart Homes." 2025 47th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). IEEE, 2025. https://ieeexplore.ieee.org/abstract/document/11253505/    

For details on data collection, please refer here: https://github.com/mmsandhu/IMU-HAR-IL    

#### The structure of this repository is as follows:    

```
har-framework  
├── helpers  
    ├── plot_cm.py  
├── neural_networks  
    ├── model  
        ├── cnn.py  
        ├── deep_conv_lstm.py  
        ├── gru.py  
        ├── callbacks  
            ├── keras_callback.py  
            ├── learning_curves.py  
            ├── utils.py  
    ├── preprocessing  
        ├── pipeline.py  
    ├── run_neural_networks.py  
├── transformers  
    ├── evaluate  
        ├── evaluate_classes.py  
    ├── preprocessing  
        ├── dataset.py  
        ├── pipeline_copy.py  
        ├── pipeline.py  
        ├── utils  
            ├── class_balancing.py  
    ├── run_transformers.py  
    ├── train  
        ├── train_tiny_transformer.py  
        ├── train_transformer_encoder.py    
        ├── models  
            ├── callbacks
                ├── learning_curves.py    
            ├── blocks.py  
            ├── tiny_transformer.py  
            ├── transformer_encoder.py    
├── config.py  
├── cross_validation_scores.py  
├── main.py    
├── README.md  
```

### Usage
The deep learning implementation allows different models to be compared:        
```
python main.py --model <model_name>
```
Available model_name options are:     
- tiny_transformer    
- transformer_encoder    
- cnn    
- gru    
- deep_conv_lstm       

The framework is separated into models bulit using Tensorflow (neural networks) and the models built using PyTorch (transformers). 

config.py can be used to change the parametrisation of the deep learning models.  

The data pipeline expects data to as a PyTorch tensor. The following tensor structure was used for this project:    
```
checkpoint = {
    "samples":
    "label_map":
    "sensor_index":
}
```

### References
The following papers and GitHub repositories were used to develop or inspire this framework. Copyright permissions and additional references have been included where required throughout the repository.     
- Tiny transformer encoder model: Lamaakal, I., Yahyati, C., Maleh, Y. et al. A tiny inertial transformer for human activity recognition via multimodal knowledge distillation and explainable AI. Sci Rep 15, 42335 (2025). https://doi.org/10.1038/s41598-025-26297-2    
- Transformer encoder model: Shavit and Klein, Boosting Inertial-based Human Activity Recognition with Transformers, IEEE Open Access (2021). https://doi.org/10.1109/ACCESS.2021.3070646       
- Inspiration for CNN model: R. Maurya, T. H. Teo, S. H. Chua, H.-C. Chow, and I.-C. Wey, “Complex Human Activities Recognition Based on High Performance 1D CNN Model,” in 2022 IEEE 15th International Symposium on Embedded Multicore/Many-core Systems-on-Chip (MCSoC) (2022). https://doi.org/10.1109/MCSoC57363.2022.00059     
- Inspiration for GRU model: J. Okai, S. Paraschiakos, M. Beekman, A. Knobbe, and C. R. de S´a, “Building robust models for Human Activity Recognition from raw accelerometers data using Gated Recurrent Units and Long Short Term Memory Neural Networks,” in 2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (2019). https://doi.ogr/10.1109/EMBC.2019.8857288    
- Inspiration for deep convolutional LSTM: V. Bijalwan, V. B. Semwal, and V. Gupta, “Wearable sensor-based pattern mining for humanactivity recognition: Deep learning approach,” The Industrial Robot, (2022). https://doi.org/10.1108/IR-09-2020-0187    
- Inspiration for deep convolutional LSTM: N. Gaud, M. Rathore, and U. Suman, “MHCNLS-HAR: Multiheaded CNN-LSTM-Based Human Activity Recognition Leveraging a Novel Wearable Edge Device for Elderly Health Care,” IEEE Sensors Journal, (2024). https://doi.org/10.1109/JSEN.2024.3450499    
- Repository implementation: Takumi Watanabe, Deep Learning (and Machine Learning) for Human Activity Recognition, https://github.com/takumiw/Deep-Learning-for-Human-Activity-Recognition/tree/master?tab=MIT-1-ov-file   
