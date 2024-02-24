# Speaker Recognition

1. Install the required packages
    ```bash
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install -r requirements.txt
    ```
2. Preprocess the data & extract the features in file `feature_extraction.py`
    ```bash
    python feature_extraction.py default.yaml
    ```
3. If current system is Windows, you need to edit `speechbrain.utils.torch_audio_backend` to ignore the warning:
    ```python
   def check_torchaudio_backend():
       """Checks the torchaudio backend and sets it to soundfile if
       windows is detected.
       """
       current_system = platform.system()
       if current_system == "Windows":
           pass
    ```

4. Train the model in file `train.py`
    ```bash
    python main.py default.yaml
    ```
5. change the `config.yaml` file to change the hyperparameters
    ```yaml
    --test_only: False # if you want to test the model
    --enable_plda: True # if you want to use PLDA
    --pca_components: 140 # number of PCA components
   ```
