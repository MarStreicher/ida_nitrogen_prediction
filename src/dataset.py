from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import List, Tuple
import pandas as pd
import numpy as np
import os

class SpectralData(Dataset):
    def __init__(
        self,
        directory_path: str = None,
        domain_list: List[str] = None,
        hsr_columns: List[str] = None,
        trait_list: List[str] = None,
    ):
        super().__init__()
        self.directory_path = directory_path
        self.domain_list = domain_list if domain_list else []
        self.hsr_columns = hsr_columns if hsr_columns else [str(wavelength) for wavelength in range(350,2501)]
        self.trait_list = trait_list if trait_list else []
        self.frame = None
        self.data = None
        self.input = None
        self.target = None
        self.input_other = None
        self.input_train = None
        self.input_test = None
        self.input_validation = None
        self.target_other = None
        self.target_train = None
        self.target_test = None
        self.target_validation = None
        
        if not self.directory_path or not os.path.exists(self.directory_path):
            raise FileNotFoundError
        
        self.frame, self.data, self.input, self.target = self._create_frame()
        self._normalize_data()
        self.input_other, self.input_train, self.input_validation, self.input_test, self.target_other, self.target_train, self.target_validation, self.target_test = self._create_train_test_split()
        
    def _create_frame(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.array, np.array]:
        frame = pd.DataFrame()
        for domain in self.domain_list:
            file_path = os.path.join(self.directory_path, f"{domain}_measured_reflectance_param.csv")
            species_frame = pd.read_csv(file_path)

            na_columns = species_frame.columns[species_frame.isna().all()]
            species_frame = species_frame.drop(na_columns, axis=1)
            
            frame = pd.concat([frame, species_frame])
            
        frame = frame.dropna()
        input_data = frame[self.hsr_columns].values
        target_data = frame[self.trait_list].values
        data = frame.values
        
        return frame, data, input_data, target_data
    
    def _normalize_data(self):
        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.input = self.input_scaler.fit_transform(self.input)
        self.target = self.target_scaler.fit_transform(self.target)
    
    def _create_train_test_split(self):
        input_other, input_test, target_other, target_test = train_test_split(self.input, self.target, test_size=0.15, random_state=42, shuffle=True) 
        input_train, input_validation, target_train, target_validation = train_test_split(input_other, target_other, test_size=0.15, random_state=42, shuffle=True) 
        
        return input_other, input_train, input_validation, input_test, target_other, target_train, target_validation, target_test,
        
    def __len__(self) -> int:
        return len(self.frame)
    
    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        instance = self.input[index]
        target = self.target[index]
        
        if not self.trait_list:
            target = self.input[index]
            
        return instance, target