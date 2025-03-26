import yaml
import functools
from typing import Callable, Optional, List, Dict, Tuple, Any
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import json
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.optim as optim
import torch_geometric.transforms
import joblib
import time
from pydantic import BaseModel, Field, field_validator
import torch.nn.functional as F
import enum
import diskcache
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys

from models import MGModel

def setup_logging(log_file= "app.log", level=logging.INFO):
    """Sets up logging to both file and stdout."""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

logger = setup_logging()

# RDKit Processing
class RDKitStep(enum.Enum):
    """Enumeration of available RDKit processing steps."""
    HYDROGENATE = "hydrogenate"
    SANITIZE = "sanitize"
    KEKULIZE = "kekulize"
    EMBED = "embed"
    OPTIMIZE = "optimize"

class Hydrogenator:
    """Adds hydrogens to molecules."""
    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        logger.debug("Hydrogenating molecule.")
        return Chem.AddHs(mol, addCoords=True)

class Sanitizer:
    """Sanitizes molecules."""
    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        logger.debug("Sanitizing molecule.")
        Chem.SanitizeMol(mol)
        return mol

class Kekulizer:
    """Kekulizes molecules."""
    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        logger.debug("Kekulizing molecule.")
        Chem.Kekulize(mol)
        return mol

class Embedder:
    """Embeds molecules using ETKDGv2."""
    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        logger.debug("Embedding molecule.")
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv2())
        return mol

class Optimizer:
    """Optimizes molecules using MMFF."""
    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        logger.debug("Optimizing molecule.")
        AllChem.MMFFOptimizeMolecule(mol)
        return mol

RDKIT_STEPS: dict[RDKitStep, Callable[[Chem.Mol], Chem.Mol]] = {
    RDKitStep.HYDROGENATE: Hydrogenator(),
    RDKitStep.SANITIZE: Sanitizer(),
    RDKitStep.KEKULIZE: Kekulizer(),
    RDKitStep.EMBED: Embedder(),
    RDKitStep.OPTIMIZE: Optimizer(),
}

class RDKitConfig(BaseModel):
    """Configuration for RDKit molecule processing."""
    steps: List[RDKitStep] = Field(
        default=[RDKitStep.HYDROGENATE, RDKitStep.SANITIZE, RDKitStep.KEKULIZE, RDKitStep.EMBED, RDKitStep.OPTIMIZE],
        description="List of RDKit processing steps.",
    )

    @field_validator("steps")
    def validate_steps(cls, v):
        """Validates the list of RDKit steps."""
        valid_steps = list(RDKitStep)
        for step in v:
            if step not in valid_steps:
                raise ValueError(f"Invalid RDKit step: {step}. Valid steps are: {[s.value for s in valid_steps]}")
        return v

class DataConfig(BaseModel):
    """Configuration for data loading."""
    root_dir: str = Field(..., description="Root directory of the dataset.")
    node_target_csv: str = Field(..., description="Path to the target CSV file.")
    use_cache: bool = Field(True, description="Whether to use cached processed data.")
    train_split: float = Field(..., description="Ratio of the dataset to use for training.")
    valid_split: float = Field(..., description="Ratio of the dataset to use for validation.")

    @field_validator("root_dir")
    def validate_root_dir(cls, v):
        """Validates the root directory."""
        if not Path(v).is_dir():
            raise ValueError(f"Root directory does not exist: {v}")
        return v

    @field_validator("node_target_csv")
    def validate_target_csv(cls, v, info: 'pydantic_core.ValidationInfo'):
        """Validates the target CSV file path."""
        root_dir = info.data['root_dir']
        target_path = Path(root_dir) / v
        if not target_path.is_file():
            raise ValueError(f"Target CSV file does not exist: {target_path}")
        return v

    @field_validator("train_split", "valid_split")
    def validate_splits(cls, v):
        """Validates the split ratios."""
        if not 0 < v < 1:
            raise ValueError("Split ratios must be between 0 and 1.")
        return v

    @field_validator("train_split")
    def validate_split_sum(cls, train_split, info: 'pydantic_core.ValidationInfo'):
        """Validates the sum of train and validation split ratios."""
        valid_split = info.data.get("valid_split")
        if valid_split is not None and train_split + valid_split >= 1:
            raise ValueError("train_split + valid_split must be less than 1.")
        return train_split

class ModelConfig(BaseModel):
    """Configuration for the model."""
    batch_size: int = Field(32, description="Batch size for training.")
    learning_rate: float = Field(0.001, description="Learning rate for the optimizer.")
    weight_decay: float = Field(0.0001, description="Weight decay for the optimizer.")
    step_size: int = Field(50, description="Step size for the learning rate scheduler.")
    gamma: float = Field(0.5, description="Gamma for the learning rate scheduler.")
    reduce_lr_factor: float = Field(0.5, description="Factor for reducing learning rate on plateau.")
    reduce_lr_patience: int = Field(10, description="Patience for reducing learning rate on plateau.")
    early_stopping_patience: int = Field(20, description="Patience for early stopping.")
    early_stopping_delta: float = Field(0.001, description="Minimum change in validation loss to qualify as an improvement.")
    l1_regularization_lambda: float = Field(0.001, description="Lambda for L1 regularization.")
    first_layer_type: Optional[str] = Field("custom_mp", description="Type of the first layer.")
    hidden_channels: int = Field(256, description="Number of hidden channels in the model.")
    second_layer_type: Optional[str] = Field("custom_mp", description="Type of the second layer.")
    dropout_rate: float = Field(0.5, description="Dropout rate in the model.")

class Config(BaseModel):
    """Main configuration class."""
    rdkit_processing: RDKitConfig = Field(default_factory=RDKitConfig, description="RDKit processing configuration.")
    data: DataConfig = Field(..., description="Data loading configuration.")
    model: ModelConfig = Field(..., description="Model training configuration.")

    @classmethod
    def from_yaml(cls, config_path: str):
        """Loads and validates configuration from a YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

# Configuration-driven RDKitMoleculeProcessor with explicit pipeline
def compose(*functions):
    """Composes multiple functions into a single pipeline."""
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

class RDKitProcessingError(Exception):
    """Base class for RDKit processing exceptions."""
    pass

class RDKitKekulizeError(RDKitProcessingError):
    """Exception for Kekulization errors."""
    pass

class RDKitMoleculeProcessor:
    """Processes RDKit molecules with a sequence of steps defined by configuration using an explicit pipeline."""

    def __init__(self, config: 'Config'):
        """Initializes the processor with the given configuration."""
        self.config = config.rdkit_processing
        self.pipeline = self._create_pipeline()
        logger.debug("RDKitMoleculeProcessor initialized.")

    def _create_pipeline(self) -> Callable[[Chem.Mol], Optional[Chem.Mol]]:
        """Creates the processing pipeline based on the configuration using a functional approach."""
        steps = [RDKIT_STEPS[step] for step in self.config.steps]
        return compose(*steps)

    def process(self, mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
        """Processes a molecule through the defined steps using the pipeline."""
        if mol is None:
            logger.warning("Attempted to process a None molecule.")
            return None
        try:
            return self.pipeline(mol)
        except Chem.rdchem.KekulizeException as e:
            logger.error(f"Kekulization error: {e}")
            raise RDKitKekulizeError(f"Kekulization error: {e}")
        except Exception as e:
            logger.error(f"Error processing molecule: {e}")
            raise RDKitProcessingError(f"Error processing molecule: {e}")

def create_configurable_rdkit_processor(config: 'Config') -> RDKitMoleculeProcessor:
    """Creates a configurable RDKit processor."""
    return RDKitMoleculeProcessor(config)

def process_molecule(mol_path:str, config: 'Config') -> Optional[Chem.Mol]:
    """Processes a molecule using a specified or configurable processor."""
    processor = create_configurable_rdkit_processor(config)
    mol = Chem.MolFromMolFile(mol_path)
    return processor.process(mol)

# Molecule Feature Extraction
class OneHotEncoder:
    """Encodes values as one-hot vectors."""
    @staticmethod
    def encode(value: int, range_size: int) -> torch.Tensor:
        """Creates a one-hot encoded tensor."""
        return F.one_hot(torch.tensor(value), num_classes=range_size).float()

class MoleculeFeatureExtractor:
    """Extracts features from molecules."""
    @staticmethod
    def get_atomic_features_one_hot(mol: Chem.Mol, feature_ranges: Dict[str, int]) -> torch.Tensor:
        """Extracts atomic features as one-hot vectors."""
        atomic_features = []
        formal_charges = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            hybridization = int(atom.GetHybridization())
            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()
            formal_charges.append(formal_charge)
            chiral_tag = int(atom.GetChiralTag())
            implicit_valence = atom.GetImplicitValence()
            num_h = atom.GetTotalNumHs()

            atom_features = [
                OneHotEncoder.encode(atomic_num, feature_ranges["atomic_nums"]),
                OneHotEncoder.encode(hybridization, feature_ranges["hybridizations"]),
                OneHotEncoder.encode(degree, feature_ranges["degrees"]),
                OneHotEncoder.encode(MoleculeFeatureExtractor.shift_formal_charge(formal_charge, formal_charges), feature_ranges["formal_charges"]),
                OneHotEncoder.encode(chiral_tag, feature_ranges["chiral_tags"]),
                OneHotEncoder.encode(implicit_valence, feature_ranges["implicit_valences"]),
                OneHotEncoder.encode(num_h, feature_ranges["num_h_list"]),
            ]
            atomic_features.append(torch.cat(atom_features))
        return torch.stack(atomic_features)

    @staticmethod
    def get_bond_features_one_hot(mol: Chem.Mol) -> torch.Tensor:
        """Extracts bond features as one-hot vectors."""
        bond_features = []
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            is_conjugated = bond.GetIsConjugated()
            is_in_ring = bond.IsInRing()
            bond_stereo = int(bond.GetStereo())

            bond_features.append(torch.tensor([
                1 if bond_type == Chem.BondType.SINGLE else 0,
                1 if bond_type == Chem.BondType.DOUBLE else 0,
                1 if bond_type == Chem.BondType.TRIPLE else 0,
                1 if bond_type == Chem.BondType.AROMATIC else 0,
                1 if is_conjugated else 0,
                1 if is_in_ring else 0,
                bond_stereo,
            ], dtype=torch.float))
        return torch.stack(bond_features) if bond_features else torch.empty(0, 7)

    @staticmethod
    def shift_formal_charge(formal_charge, formal_charges):
        """Shifts formal charges to ensure non-negative values for one-hot encoding."""
        min_formal_charge = min(formal_charges)
        offset = abs(min_formal_charge) if min_formal_charge < 0 else 0
        return formal_charge + offset

# Molecule Processor
class MoleculeProcessor:
    """Processes molecules into graph data."""
    def __init__(self, root: str, directed: bool = False, rdkit_config: Config = None):
        """Initializes the processor with root directory, directionality, and RDKit configuration."""
        self.root = root
        self.directed = directed
        self.raw_loader = RawDataLoader(root)
        self.data_handler = ProcessedDataHandler(root)
        self.feature_ranges = self.data_handler.load_feature_ranges(self.raw_loader.load_raw_files())
        self.rdkit_config = rdkit_config
        logger.debug(f"MoleculeProcessor initialized with root: {root}, directed: {directed}")

    def load_molecule(self, mol_path: str) -> Optional[Chem.Mol]:
        """Loads a molecule from a file path."""
        logger.debug(f"Loading molecule from {mol_path}")
        return process_molecule(mol_path, config=self.rdkit_config)

    def process_molecule(self, mol: Chem.Mol) -> Optional[torch_geometric.data.Data]:
        """Processes a molecule into a graph data object."""
        logger.debug("Processing molecule to graph.")
        return self.mol_to_graph(mol, self.feature_ranges, self.directed)

    @staticmethod
    def mol_to_graph(mol: Chem.Mol, feature_ranges: Dict[str, int], directed: bool = False) -> Optional[torch_geometric.data.Data]:
        """Converts an RDKit molecule to a PyTorch Geometric Data object."""
        if mol is None:
            logger.warning("Attempted to convert a None molecule to graph.")
            return None
        x = MoleculeFeatureExtractor.get_atomic_features_one_hot(mol, feature_ranges)
        edge_attr_onehot = MoleculeFeatureExtractor.get_bond_features_one_hot(mol)
        edge_index_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index_list.append([i, j])
            if not directed:
                edge_index_list.append([j, i])
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        pos = mol.GetConformer().GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)
        graph = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr_onehot, pos=pos)
        return graph

# Data Loading and Processing
class RawDataLoader:
    """Loads raw molecule files."""
    def __init__(self, root: str):
        """Initializes the loader with the root directory."""
        self.root = Path(root)
        self.mol_dir = self.root / 'Mols'
        logger.debug(f"RawDataLoader initialized with root: {root}")

    def load_raw_files(self) -> List[str]:
        """Loads raw MOL file paths."""
        files = [str(f) for f in self.mol_dir.glob('*.mol')]
        logger.debug(f"Loaded {len(files)} raw files.")
        return files

class FeatureRangeCalculator:
    """Calculates feature ranges from molecule data."""
    @staticmethod
    def calculate_feature_ranges(mol_paths: List[str]) -> Dict[str, int]:
        """Calculates feature ranges based on the provided molecule file paths."""
        data = []
        formal_charges = []

        for mol_path in mol_paths:
            mol = Chem.MolFromMolFile(mol_path)
            if mol is not None:
                for atom in mol.GetAtoms():
                    data.append({
                        "atomic_num": atom.GetAtomicNum(),
                        "hybridization": int(atom.GetHybridization()),
                        "degree": atom.GetDegree(),
                        "formal_charge": atom.GetFormalCharge(),
                        "chiral_tag": int(atom.GetChiralTag()),
                        "implicit_valence": atom.GetImplicitValence(),
                        "num_h": atom.GetTotalNumHs()
                    })
                    formal_charges.append(atom.GetFormalCharge())

        df = pd.DataFrame(data)

        feature_ranges = {
            "atomic_nums": int(df["atomic_num"].max()) + 1 if not df.empty else 1,
            "hybridizations": int(df["hybridization"].max()) + 1 if not df.empty else 1,
            "degrees": int(df["degree"].max()) + 1 if not df.empty else 1,
            "formal_charges": max(formal_charges) + abs(min(formal_charges)) + 1 if formal_charges else 1,
            "chiral_tags": int(df["chiral_tag"].max()) + 1 if not df.empty else 1,
            "implicit_valences": int(df["implicit_valence"].max()) + 1 if not df.empty else 1,
            "num_h_list": int(df["num_h"].max()) + 1 if not df.empty else 1,
        }
        logger.debug(f"Calculated feature ranges: {feature_ranges}")
        return feature_ranges

class ProcessedDataHandler:
    """Handles saving and loading of processed data."""
    def __init__(self, root: str, use_cache: bool = True, cache_expiry: int = 3600):
        """Initializes the handler with root directory, cache settings, and cache expiry."""
        self.root = Path(root)
        self.use_cache = use_cache
        self.cache_dir = self.root / 'processed_graphs_cache'
        self.processed_dir = self.root / 'processed'
        self.feature_ranges_path = self.root / 'feature_ranges.json'
        self.cache_expiry = cache_expiry
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache = diskcache.Cache(self.cache_dir)  # Initialize diskcache
        else:
            self.cache = None
        os.makedirs(self.processed_dir, exist_ok=True)
        logger.debug(f"ProcessedDataHandler initialized with root: {root}, use_cache: {use_cache}")

    def save_feature_ranges(self, feature_ranges: Dict[str, int]) -> None:
        """Saves feature ranges to a JSON file."""
        with open(self.feature_ranges_path, 'w') as f:
            json.dump(feature_ranges, f)
        logger.debug(f"Saved feature ranges to {self.feature_ranges_path}")

    def load_feature_ranges(self, raw_file_names: List[str]) -> Dict[str, int]:
        """Loads feature ranges from a JSON file or calculates them if not present."""
        if os.path.exists(self.feature_ranges_path):
            with open(self.feature_ranges_path, 'r') as f:
                ranges = json.load(f)
                logger.debug(f"Loaded feature ranges from {self.feature_ranges_path}")
                return ranges
        else:
            feature_ranges = FeatureRangeCalculator.calculate_feature_ranges(raw_file_names)
            self.save_feature_ranges(feature_ranges)
            return feature_ranges

    def save_graph(self, graph: torch_geometric.data.Data, file_name: str) -> None:
        """Saves a graph data object to file or cache."""
        processed_path = self.processed_dir / file_name
        if self.use_cache and self.cache:
            self.cache.set(file_name, graph, expire=self.cache_expiry)
            logger.debug(f"Saved graph to cache: {file_name}")
        else:
            torch.save(graph, processed_path)
            logger.debug(f"Saved graph to file: {processed_path}")

    def load_graph(self, file_name: str) -> Optional[torch_geometric.data.Data]:
        """Loads a graph data object from file or cache."""
        processed_path = self.processed_dir / file_name
        if self.use_cache and self.cache:
            graph = self.cache.get(file_name)
            if graph is not None:
                logger.debug(f"Loaded graph from cache: {file_name}")
                return graph
            else:
                logger.info(f"Cache miss for {file_name}, recalculating.")
                return None
        else:
            try:
                torch.serialization.add_safe_globals([
                    torch_geometric.data.data.DataEdgeAttr,
                    torch_geometric.data.data.DataTensorAttr,
                    torch_geometric.data.storage.GlobalStorage
                ])
                graph = torch.load(processed_path)
                logger.debug(f"Loaded graph from file: {processed_path}")
                return graph
            except FileNotFoundError:
                logger.warning(f"Graph file not found: {processed_path}")
                return None

class DatasetError(Exception):
    '''Base exception for dataset related errors.'''
    pass

class EmptyDatasetError(DatasetError):
    '''Raised when the loaded CSV file is empty.'''
    pass

class ColumnNotFoundError(DatasetError):
    '''Raised when a specified column is not found in the CSV.'''
    def __init__(self, column_name: str) -> None:
        """Initialize with the missing column name."""
        super().__init__(f"Column '{column_name}' not found in CSV.")

class DataProcessor:
    """Processes molecule data and converts it to graph data."""
    def __init__(self, root: str, use_cache: bool = True, rdkit_config: Config = None):
        """Initializes the processor with root directory, cache settings, and RDKit configuration."""
        self.root = root
        self.use_cache = use_cache
        self.raw_loader = RawDataLoader(root)
        self.data_handler = ProcessedDataHandler(root, use_cache)
        self.molecule_processor = MoleculeProcessor(root, rdkit_config=rdkit_config)
        self.rdkit_config = rdkit_config
        logger.debug(f"DataProcessor initialized with root: {root}, use_cache: {use_cache}")

    def _load_molecule_data(self, raw_path: str) -> Optional[Chem.Mol]:
        """Loads a molecule from a file path."""
        logger.debug(f"Loading molecule data from {raw_path}")
        return self.molecule_processor.load_molecule(raw_path)

    def _process_molecule_graph(self, mol: Chem.Mol, mol_name: str, processed_name: str, node_target_df: pd.DataFrame) -> None:
        """Processes a molecule into a graph data object and saves it."""
        if mol is not None:
            graph = self.molecule_processor.process_molecule(mol)
            if graph is not None:
                try:
                    node_targets = []
                    for atom in mol.GetAtoms():
                        atom_index = atom.GetIdx()
                        node_target = node_target_df.loc[(mol_name, atom_index,)].values
                        node_targets.append(node_target)
                    node_targets = np.array(node_targets)    
                    node_targets = torch.tensor(node_targets, dtype=torch.float)
                    graph.y = node_targets
                    self.data_handler.save_graph(graph, processed_name)
                    logger.debug(f"Processed and saved graph for {mol_name}")
                except KeyError as e:
                    logger.error(f'Molecule name {mol_name} not found or index error in target file: {e}')
                    raise ColumnNotFoundError(f'Molecule name {mol_name} not found or index error in target file: {e}')
                except Exception as e:
                    logger.error(f'Error saving graph {processed_name}: {e}')
                    raise DatasetError(f'Error saving graph {processed_name}: {e}')
            else:
                logger.warning(f'Trouble processing graph {processed_name} or empty graph')
        else:
            logger.warning(f'MOL file {mol_name}.mol not found or corrupted, or rdkit processing failed')

    def process_data(self, node_target_df: pd.DataFrame) -> None:
        """Processes all molecule data and saves the corresponding graph data objects."""
        raw_file_names = self.raw_loader.load_raw_files()
        if not raw_file_names:
            logger.error("No MOL files found in the specified directory.")
            raise EmptyDatasetError("No MOL files found in the specified directory.")
        for raw_path, processed_name in zip(raw_file_names, [Path(f).name.replace('.mol', '.pt') for f in raw_file_names]):
            mol_name = Path(raw_path).name.replace('.mol', '')
            if self.use_cache and self.data_handler.cache and self.data_handler.cache.get(processed_name):
                logger.debug(f"Graph {processed_name} found in cache, skipping processing.")
                continue
            mol = self._load_molecule_data(raw_path)
            self._process_molecule_graph(mol, mol_name, processed_name, node_target_df)

class MGDataset(torch_geometric.data.Dataset):
    """Dataset for molecule graph data."""
    def __init__(self, root, directed=False, transform=None, node_target_csv='targets_n_reg.csv', use_cache=True, rdkit_config: Config = None):
        """Initializes the dataset with root directory, directionality, transform, target CSV, cache settings, and RDKit configuration."""
        self.directed = directed
        try:
            self.node_target_df = pd.read_csv(Path(root) / node_target_csv, index_col=['MoleculeName', 'atom_index',])
            logger.info("DataFrame loaded successfully.")
            if self.node_target_df.empty:
                logger.error("Target CSV file is empty.")
                raise EmptyDatasetError("Target CSV file is empty.")
        except FileNotFoundError:
            logger.error(f"Target CSV file not found: {target_csv}")
            raise DatasetError(f"Target CSV file not found: {target_csv}")
        except Exception as e:
            logger.error(f"Error loading target CSV: {e}")
            raise DatasetError(f"Error loading target CSV: {e}")
        self.use_cache = use_cache
        self.root = root
        self.data_processor = DataProcessor(root, use_cache, rdkit_config)  # DataProcessor is now Created here.
        self.pre_transform = torch_geometric.transforms.Compose([torch_geometric.transforms.NormalizeFeatures(), torch_geometric.transforms.AddSelfLoops(), torch_geometric.transforms.Distance()])
        super().__init__(root, transform)
        self.transform_list = torch_geometric.transforms.Compose([torch_geometric.transforms.RandomRotate(degrees=180), torch_geometric.transforms.RandomScale((0.9, 1.1)), torch_geometric.transforms.RandomJitter(0.01), torch_geometric.transforms.RandomFlip(0), ])
        self.data_processor.process_data(self.node_target_df)  # Process data is now called here, and the node_target_df is passed.
        self.data_list = self._load_and_transform_data()
        logger.debug(f"MGDataset initialized with root: {root}, directed: {directed}, use_cache: {use_cache}")

    def _load_and_transform_data(self):
        """Loads and transforms graph data objects."""
        data_list = []
        for processed_file_name in self.processed_file_names:
            data = self.data_processor.data_handler.load_graph(processed_file_name)
            if data is not None:
                data = self.transform_list(data)
                data_list.append(data)
            else:
                logger.warning(f"Skipping molecule {processed_file_name} due to missing data.")
        logger.debug(f"Loaded and transformed {len(data_list)} data points.")
        return data_list

    @property
    def raw_file_names(self):
        """Returns a list of raw file names."""
        return self.data_processor.raw_loader.load_raw_files()

    @property
    def processed_file_names(self):
        """Returns a list of processed file names."""
        return [Path(f).name.replace('.mol', '.pt') for f in self.raw_file_names]

    def download(self):
        """Placeholder for download functionality."""
        pass

    def process(self):
        """Placeholder for processing functionality."""
        pass  # process data is now done in the init function, and in the DataProcessor class.

    def len(self):
        """Returns the length of the dataset."""
        return len(self.data_list)

    def get(self, idx):
        """Returns the data object at the specified index."""
        return self.data_list[idx]

    def get_molecule_name(self, idx: int) -> str:
        """Returns the molecule name at the specified index."""
        processed_file_name = self.processed_file_names[idx]
        return Path(processed_file_name).stem

# Early Stopping
class EarlyStopping:
    """Early stopping mechanism."""
    def __init__(self, patience=7, verbose=False, delta=0.0, path='chk_learn.pt'):
        """Initializes the early stopping mechanism with patience, verbosity, delta, and path."""
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        logger.debug(f"EarlyStopping initialized with patience: {patience}, delta: {delta}")

    def __call__(self, valid_loss, model):
        """Checks if early stopping should be triggered."""
        if self.best_score is None:
            self.best_score = valid_loss
            self.save_model_state(model)
        elif valid_loss < self.best_score - self.delta:
            if self.verbose:
                logger.info(f'Validation Loss improves {self.best_score:.4f}->{valid_loss:.4f}=>$ave model')
            self.best_score = valid_loss
            self.save_model_state(model)
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f'Machine Learning patience ticks {self.counter} from {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_model_state(self, model):
        """Saves the model state to the specified path."""
        torch.save(model.state_dict(), self.path)
        logger.debug(f"Saved model state to {self.path}")

# Configuration Loading Module
def load_config(config_path: str) -> 'Config':
    """Loads and validates configuration from a YAML file."""
    try:
        config = Config.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Error loading configuration: {e}")
        raise (f"Error loading configuration: {e}")


def split_dataset(dataset: torch_geometric.data.Dataset, train_ratio: float, valid_ratio: float, node_target_df: pd.DataFrame):
    """
    Splits a PyTorch Geometric dataset into train, validation, and test sets,
    with stratification based on node_target values.
    """

    indices = list(node_target_df.index) # change indices to be the index of the dataframe.
    target_values = [node_target_df.loc[idx].values[0] for idx in indices]  # Access using loc and the correct index.
    target_series = pd.Series(target_values)

    # Binning for regression tasks
    bins = pd.qcut(target_series, q=10, labels=False, duplicates='drop')  # Adjust 'q' as needed.
    # If classification, use target_values directly: bins = target_series

    train_indices, temp_indices, _, temp_bins = train_test_split(
        indices, bins, train_size=train_ratio, stratify=bins, random_state=42
    )
    valid_indices, test_indices, _, _ = train_test_split(
        temp_indices, temp_bins, train_size=valid_ratio / (1 - train_ratio), stratify=temp_bins, random_state=42
    )

    train_dataset = torch.utils.data.Subset(dataset, [dataset.processed_file_names.index(mol_name + ".pt") for mol_name, atom_index in train_indices])
    valid_dataset = torch.utils.data.Subset(dataset, [dataset.processed_file_names.index(mol_name + ".pt") for mol_name, atom_index in valid_indices])
    test_dataset = torch.utils.data.Subset(dataset, [dataset.processed_file_names.index(mol_name + ".pt") for mol_name, atom_index in test_indices])

    logger.info(f"Dataset split into train ({len(train_dataset)}), validation ({len(valid_dataset)}), and test ({len(test_dataset)}) sets.")
    return train_dataset, valid_dataset, test_dataset

# Training Loop Class
class TrainingLoop:
    """Encapsulates the training loop logic."""
    def __init__(self, model, criterion, optimizer, step_lr, device, l1_lambda):
        """Initializes the training loop with model, criterion, optimizer, learning rate scheduler, device, and L1 regularization lambda."""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.step_lr = step_lr
        self.device = device
        self.l1_lambda = l1_lambda
        logger.debug("TrainingLoop initialized.")

    def train_epoch(self, train_loader):
        """Trains the model for one epoch."""
        self.model.train()
        total_loss = 0
        num_nodes = 0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            node_out, l1_reg = self.model(batch.x, batch.edge_index, batch.batch)
            node_target = batch.y
            node_loss = self.criterion(node_out, node_target)
            node_loss += l1_reg * self.l1_lambda
            node_loss.backward()
            self.optimizer.step()
            total_loss += node_loss.item() * batch.num_nodes
            num_nodes += batch.num_nodes
        self.step_lr.step()
        avg_loss = total_loss / num_nodes
        logger.debug(f"Training Epoch Loss: {avg_loss:.4f}")
        return avg_loss

    def validate_epoch(self, valid_loader):
        """Validates the model for one epoch."""
        self.model.eval()
        total_loss = 0
        num_nodes = 0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(self.device)
                node_out, _ = self.model(batch.x, batch.edge_index, batch.batch)
                node_target = batch.y
                node_loss = self.criterion(node_out, node_target)
                total_loss += node_loss.item() * batch.num_nodes
                num_nodes += batch.num_nodes
                all_targets.append(node_target.cpu().numpy())
                all_predictions.append(node_out.cpu().numpy())

        avg_loss = total_loss / num_nodes
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        mae = mean_absolute_error(all_targets, all_predictions)
        mse = mean_squared_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        explained_variance = explained_variance_score(all_targets, all_predictions)
        logger.debug(f"Validation Epoch Loss: {avg_loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}, Explained Variance: {explained_variance:.4f}")
        return avg_loss, mae, mse, r2, explained_variance

    def test_epoch(self, test_loader, return_predictions=False):
        """Tests the model for one epoch."""
        self.model.eval()
        total_loss = 0
        num_nodes = 0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                node_out, _ = self.model(batch.x, batch.edge_index, batch.batch)
                node_target = batch.y
                node_loss = self.criterion(node_out, node_target)
                total_loss += node_loss.item() * batch.num_nodes
                num_nodes += batch.num_nodes
                all_targets.append(node_target.cpu().numpy())
                all_predictions.append(node_out.cpu().numpy())

        avg_loss = total_loss / num_nodes
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        mae = mean_absolute_error(all_targets, all_predictions)
        mse = mean_squared_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        explained_variance = explained_variance_score(all_targets, all_predictions)
        logger.info(f"Test Epoch Loss: {avg_loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}, Explained Variance: {explained_variance:.4f}")
        if return_predictions:
            return avg_loss, mae, mse, r2, explained_variance, all_targets, all_predictions
        return avg_loss, mae, mse, r2, explained_variance

# Trainer Class
class Trainer:
    """Manages the training and validation process."""
    def __init__(self, model, criterion, optimizer, step_lr, red_lr, early_stopping, config, device):
        """Initializes the trainer with model, criterion, optimizer, learning rate schedulers, early stopping, configuration, and device."""
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.step_lr = step_lr
        self.red_lr = red_lr
        self.early_stopping = early_stopping
        self.config = config
        self.device = device
        self.training_loop = TrainingLoop(self.model, self.criterion, self.optimizer, self.step_lr, self.device, self.config.model.l1_regularization_lambda)
        logger.debug("Trainer initialized.")

    def train_and_validate(self, train_loader, valid_loader):
        """Trains and validates the model over multiple epochs."""
        train_losses, valid_losses, maes, mses, r2s, explained_variances = [], [], [], [], [], []
        for epoch in range(self.config.model.early_stopping_patience * 2):
            train_loss = self.training_loop.train_epoch(train_loader)
            valid_loss, mae, mse, r2, explained_variance = self.training_loop.validate_epoch(valid_loader)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            maes.append(mae)
            mses.append(mse)
            r2s.append(r2)
            explained_variances.append(explained_variance)

            self.red_lr.step(valid_loss)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                logger.info("Early stopping triggered.")
                break
        return train_losses, valid_losses, maes, mses, r2s, explained_variances

    def test_epoch(self, test_loader, return_predictions=False):
        """Tests the model for one epoch."""
        return self.training_loop.test_epoch(test_loader, return_predictions)

class Plot:
    """Handles plotting of training and validation metrics."""
    @staticmethod
    def plot_losses(train_losses, valid_losses):
        """Plots training and validation losses."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(valid_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_metrics_vs_epoch(maes, mses, r2s, explained_variances):
        """Plots metrics (MAE, MSE, R2, Explained Variance) vs. epoch."""
        epochs = range(1, len(maes) + 1)

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(epochs, maes, label='MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('MAE vs. Epoch')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(epochs, mses, label='MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE vs. Epoch')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(epochs, r2s, label='R2')
        plt.xlabel('Epoch')
        plt.ylabel('R2')
        plt.title('R2 vs. Epoch')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(epochs, explained_variances, label='Explained Variance')
        plt.xlabel('Epoch')
        plt.ylabel('Explained Variance')
        plt.title('Explained Variance vs. Epoch')
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    from pathlib import Path
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
    from torch_geometric.loader import DataLoader
    import numpy as np

    data_dir = Path('C:/Chem_Data')

    config_path = data_dir / 'config.yaml'
    config = load_config(config_path)

    dataset = MGDataset(root=config.data.root_dir, node_target_csv=config.data.node_target_csv, use_cache=config.data.use_cache, rdkit_config=config)

    in_channels = dataset[0].x.shape[1]
    out_channels = dataset.node_target_df.shape[1]

    torch.manual_seed(11)

    train_dataset, valid_dataset, test_dataset = split_dataset(dataset, config.data.train_split, config.data.valid_split, dataset.node_target_df)

    train_loader = DataLoader(train_dataset, batch_size=config.model.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.model.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.model.batch_size, shuffle=False)

    model = MGModel(
        in_channels=in_channels,
        out_channels=out_channels,
        first_layer_type=config.model.first_layer_type,
        second_layer_type=config.model.second_layer_type,
        hidden_channels=config.model.hidden_channels,
        dropout_rate=config.model.dropout_rate,
        gat_heads=1,
        transformer_heads=1,
    )

    logger.info(f'Model Architecture {model}')

    criterion = nn.HuberLoss(reduction='mean', delta=0.1)
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay)

    step_lr = StepLR(optimizer, step_size=config.model.step_size, gamma=config.model.gamma)
    red_lr = ReduceLROnPlateau(optimizer, mode='min', factor=config.model.reduce_lr_factor, patience=config.model.reduce_lr_patience)

    early_stopping = EarlyStopping(patience=config.model.early_stopping_patience, verbose=True, delta=config.model.early_stopping_delta)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    trainer = Trainer(model, criterion, optimizer, step_lr, red_lr, early_stopping, config, device)
    train_losses, valid_losses, maes, mses, r2s, explained_variances = trainer.train_and_validate(train_loader, valid_loader)
    test_loss, test_mae, test_mse, test_r2, test_explained_variance, test_targets, test_predictions = trainer.test_epoch(test_loader, return_predictions=True)

    # Save test targets and predictions
    np.save('test_targets.npy', np.array(test_targets))
    np.save('test_predictions.npy', np.array(test_predictions))

    logger.info(f'Test Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, R2: {test_r2:.4f}, Explained Variance: {test_explained_variance:.4f}')

    # Plotting
    Plot.plot_losses(train_losses, valid_losses)
    Plot.plot_metrics_vs_epoch(maes, mses, r2s, explained_variances)


    
