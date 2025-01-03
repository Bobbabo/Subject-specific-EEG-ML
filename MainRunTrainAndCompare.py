import matplotlib.pyplot as plt
import mne
import seaborn as sns
import torch
from braindecode import EEGClassifier
from sklearn.pipeline import make_pipeline
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.dataset import ValidSplit
import matplotlib.pyplot as plt
import pandas as pd
import os

from moabb.datasets import BNCI2014_001, BNCI2014_004
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import MotorImagery
from moabb.utils import setup_seed


from moabb_Private_Encoder_Thesis.moabb.evaluations import SubjectParamEvaluation
from shallow import CollapsedShallowNet ,ShallowFBCSPNet
from shallowDict import ShallowPrivateTemporalDictNetSlow, ShallowPrivateSpatialDictNetSlow, ShallowPrivateCollapsedDictNetSlow, SubjectDicionaryFCNet

mne.set_log_level(False)

print(f"Torch Version: {torch.__version__}")

# Set up GPU if it is there
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
print("GPU is", "AVAILABLE" if cuda else "NOT AVAILABLE")

seed = 142
setup_seed(seed)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameter

# learning rate 1e-4

# batch = 2^7
LEARNING_RATE = 0.0001  # parameter taken from Braindecode
WEIGHT_DECAY = 0  # parameter taken from Braindecode
BATCH_SIZE = 128  # parameter taken from BrainDecode
EPOCH = 3000
PATIENCE = 100
fmin = 4
fmax = 100
tmin = 0
tmax = None


dataset = BNCI2014_001()
paradigm = MotorImagery(
    fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax
)

X, _, _ = paradigm.get_data(dataset=dataset)

subjects = dataset.subject_list
        

def make_classifier(module):
    clf = EEGClassifier(
        module=module,  
        module__n_chans=X.shape[1],  # number of input channels
        module__n_outputs=len(dataset.event_id),  # number of output classes
        module__n_times=X.shape[2],  # length of the input signal in time points
        optimizer=torch.optim.Adam,
        optimizer__lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=EPOCH,
        warm_start=True, #To keep training the model further for each fit instead of re-initializing
        train_split=ValidSplit(0.2, random_state=seed),
        device=device,
        callbacks=[
            #EarlyStopping(monitor="valid_loss", patience=PATIENCE),
            EpochScoring(
                scoring="accuracy", on_train=True, name="train_acc", lower_is_better=False
            ),
            EpochScoring(
                scoring="accuracy", on_train=False, name="valid_acc", lower_is_better=False
            ),
        ],
        verbose=1,
    )
    
    return clf


clf1 = make_classifier(SubjectDicionaryFCNet)

clf2 = make_classifier(CollapsedShallowNet)

clf3 = make_classifier(ShallowPrivateTemporalDictNetSlow)

clf4 = make_classifier(ShallowFBCSPNet)

clf5 = make_classifier(ShallowPrivateSpatialDictNetSlow)

clf6 = make_classifier(ShallowPrivateCollapsedDictNetSlow)

# Create a pipeline with the classifier

pipes = { "SubjectDicionaryFCNet": make_pipeline(clf1), "CollapsedShallowNet": make_pipeline(clf2), "ShallowPrivateTemporalDictNetSlow": make_pipeline(clf3), "ShallowFBCSPNet": make_pipeline(clf4), "ShallowPrivateSpatialDictNetSlow": make_pipeline(clf5), "ShallowPrivateCollapsedDictNetSlow": make_pipeline(clf6)}

results_list = []
# Ensure the output directory exists
output_dir = f"./results_{seed}_{dataset.code}"
os.makedirs(output_dir, exist_ok=True)

# Modify plot and data saving within the loop
for pipe_name, pipe in pipes.items():
    unique_suffix = f"{pipe_name}_braindecode_example"
    
    evaluation = SubjectParamEvaluation(
        paradigm=paradigm,
        datasets=[dataset],
        suffix=unique_suffix,
        overwrite=True,
        return_epochs=True,
        random_state=seed,
        n_jobs=1,
        hdf5_path=f"{output_dir}/{pipe_name}",
        save_model=True
    )
    
    # Run the evaluation process for this pipeline
    results = evaluation.process({pipe_name: pipe})

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/{pipe_name}_results.csv", index=False)

    # Save individual bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, y="score", x="subject", palette="viridis")
    plt.title(f"Model Performance by Subject - {pipe_name}")
    plt.ylabel("Score")
    plt.xlabel("Subject")
    plt.savefig(f"{output_dir}/{pipe_name}_performance.png")
    plt.close()
    
    results_list.append(results_df)

# Concatenate all results
results_all = pd.concat(results_list)
results_all.to_csv(f"{output_dir}/all_results.csv", index=False)

# Save combined bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=results_all, x="subject", y="score", hue="pipeline", palette="viridis")
plt.title("Scores per Subject for Each Pipeline")
plt.xlabel("Subject")
plt.ylabel("Score")
plt.legend(title="Pipeline")
plt.savefig(f"{output_dir}/combined_performance.png")
plt.close()