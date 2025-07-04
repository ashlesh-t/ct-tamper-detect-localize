# 🚀 CT-GAN for Tampered Lung CT Scan Generation (Kaggle Version)

**PyTorch adaptation of the original Keras CT-GAN project by**  
*Yisroel Mirsky, Tom Mahler, Ilan Shelef, Yuval Elovici.*

**📌 This README is tailored for training on Kaggle Notebooks.**

---

## ✨ What is this?

CT-GAN is a GAN model that learns to *inject* (add) and *remove* simulated lung tumors in 3D CT scans. This implementation allows researchers to generate synthetic medical data for training and testing purposes.

---

## 📦 Why Kaggle-specific?

- ✅ Kaggle limits dataset upload size (~20GB)
- ✅ Medical CT datasets are huge (60–100GB)
- ✅ So **we split** them into 4+ chunks (~16–18GB each)
- ✅ Then we train on Kaggle using these chunks

---

## 🗺️ Full Pipeline Steps

1. **Split data into chunks**
2. **Build CSV metadata** (coordinates of tumors or healthy regions)
3. **Extract training data (.npy) from chunks**
4. **Upload .npy chunks to Kaggle Datasets**
5. **Attach datasets in Kaggle Notebook**
6. **Clone this repo**
7. **Check and edit `parameters.yml`**
8. **Train models: Injector and Remover**

---

## ⚙️ 1️⃣ Prepare Data Locally

### 1.1 Split Your DICOM Data

Organize *each chunk* like:

```
/chunk1/main-folder/dcm-id1/*.dcm
                   dcm-id2/*.dcm
                   ...
coords.csv
```

- ✅ Recommended size: ~16–18GB per chunk
- ✅ Typically ~4 chunks for 60GB total data

### 1.2 Build `coords.csv` for Each Chunk

Required columns:
```
filename,z,x,y
```

**filename**: path *within your main-folder* down to dcm-id folder.

Example format:
```
main-folder/dcm-id/
```

**z,x,y**: center of tumor in voxel coordinates.

#### 📌 For Tumor (Unhealthy) Data

- Usually comes with metadata (e.g. BraTS centroid annotations)
- Example row:
```
patient1/ID123,45,68,59
```

#### 📌 For Healthy Data

- No true tumor
- Suggested approach:
  - Use lung segmentation model (e.g. MONAI / Hugging Face)
  - Detect lung edges
  - Randomly pick plausible tumor-like locations *inside lungs*
  - Add them to CSV as pseudo-centroids

- ✅ Example row:
```
normal/ID999,50,65,60
```

---

## ⚙️ 2️⃣ Run Extractor Scripts Locally

You *must* run **on each chunk separately**.

```bash
python3 1A_build_injector_trainset.py
python3 1D_build_remover_trainset.py
```

**Note**: Remember to change parameters in `parameters.yml` when building on local machine:

```yaml
samples_file: "D:/YourPath/output/healthy_samples.npy"
```

### ✅ 2.1 Extract Injector Data (Unhealthy)

For *each chunk*:

```bash
python 1A_build_injector_trainset.py --chunk-path "path/to/chunk1"
```

✅ Produces:
```
chunk1/unhealthy_samples.npy
```

Do for all chunks.

### ✅ 2.2 Extract Remover Data (Healthy)

```bash
python 1B_build_remover_trainset.py --chunk-path "path/to/chunk1"
```

✅ Produces:
```
chunk1/healthy_samples.npy
```

Do for all chunks.

### ✅ Resulting Folder Example

```
/chunks/
    chunk1/
        unhealthy_samples.npy
        healthy_samples.npy
    chunk2/
        unhealthy_samples.npy
        healthy_samples.npy
    ...
```

✅ Upload these .npy files to Kaggle as a dataset.

---

## ⚙️ 3️⃣ Upload to Kaggle

1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets)
2. Create New Dataset
3. Add all .npy chunks
4. Make it Private or Public

---

## ⚙️ 4️⃣ Create a Kaggle Notebook

- ✅ Click New Notebook
- ✅ Attach your dataset (with .npy chunks)
- ✅ Turn on:
  - Internet (to clone repo)
  - Persistence (to keep models saved)

---

## ⚙️ 5️⃣ Clone This Repo in Kaggle

Add this cell in your notebook:

```python
!git clone -b generate-tampered https://github.com/ASHLESHA05/Detect-and-Locate-Tampered-Medical-Images.git
%cd Detect-and-Locate-Tampered-Medical-Images
```

✅ Replace with your repo URL if different.

---

## ⚙️ 6️⃣ Check parameters.yml

You need to make sure your .npy paths match your Kaggle Dataset mount.

✅ First, print it:

```python
with open("parameters.yml") as f:
    print(f.read())
```

### ✅ Example parameters.yml

You will see something like:

```yaml
healthy:
  chunks:
    - /kaggle/input/your-dataset/chunk1/healthy_samples.npy
    - /kaggle/input/your-dataset/chunk2/healthy_samples.npy
    # ...
unhealthy:
  chunks:
    - /kaggle/input/your-dataset/chunk1/unhealthy_samples.npy
    # ...
```

✅ Adjust paths if needed!

### ✅ Edit parameters.yml

Example cell:

```python
new_params = """
healthy:
  chunks:
    - /kaggle/input/my-dataset/chunk1/healthy_samples.npy
    - /kaggle/input/my-dataset/chunk2/healthy_samples.npy
    - /kaggle/input/my-dataset/chunk3/healthy_samples.npy
    - /kaggle/input/my-dataset/chunk4/healthy_samples.npy

unhealthy:
  chunks:
    - /kaggle/input/my-dataset/chunk1/unhealthy_samples.npy
    - /kaggle/input/my-dataset/chunk2/unhealthy_samples.npy
    - /kaggle/input/my-dataset/chunk3/unhealthy_samples.npy
    - /kaggle/input/my-dataset/chunk4/unhealthy_samples.npy

modelpath_inject: "./models/INJ"
modelpath_remove: "./models/REM"
"""

with open("parameters.yml", "w") as f:
    f.write(new_params)
```

✅ Verify by printing again:

```python
with open("parameters.yml") as f:
    print(f.read())
```

---

## ⚙️ 7️⃣ Train the Injector Model

✅ Add this cell in notebook:

```bash
!python 2A_train_injector.py
```

Saves trained injector GAN to `modelpath_inject`.

After training completes, download those model files.

---

## ⚙️ 8️⃣ (Optional) Reset Kaggle Memory

✅ You can restart the Kaggle session to clear memory.

---

## ⚙️ 9️⃣ Train the Remover Model

- ✅ Create new notebook or restart session
- ✅ Attach same dataset
- ✅ Clone repo again:

```python
!git clone https://github.com/YOUR_USERNAME/ct-gan.git
%cd ct-gan
```

- ✅ Make sure `parameters.yml` is correct
- ✅ Train:

```bash
!python 2B_train_remover.py
```

- ✅ Saves trained remover GAN to `modelpath_remove`
- ✅ Download those models too

---

## ⚙️ ✅ Notes

- Make sure you enable **Persistence** in Kaggle to save outputs between runs
- Adjust `parameters.yml` for different chunk counts or paths
- You can use your own local `parameters_local.yml` if you want different config

---

## 🧭 Example Complete Kaggle Notebook Cells

```python
# Clone repo
!git clone https://github.com/YOUR_USERNAME/ct-gan.git
%cd ct-gan

# Check parameters
with open("parameters.yml") as f:
    print(f.read())

# (Optional) Update parameters
new_params = """
healthy:
  chunks:
    - /kaggle/input/my-dataset/chunk1/healthy_samples.npy
    # ... add more chunks
unhealthy:
  chunks:
    - /kaggle/input/my-dataset/chunk1/unhealthy_samples.npy
    # ... add more chunks
modelpath_inject: "./models/INJ"
modelpath_remove: "./models/REM"
"""
with open("parameters.yml", "w") as f:
    f.write(new_params)

# Train injector
!python 2A_train_injector.py

# Train remover
!python 2B_train_remover.py
```

---

## ✅ Credits

- Yisroel Mirsky, Tom Mahler, Ilan Shelef, Yuval Elovici
- [Original Paper](https://arxiv.org/abs/1901.03597)

---

## ✅ License

MIT. See LICENSE file for details.