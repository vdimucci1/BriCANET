# BriCANet Classifier - GUI Application 🧠

Welcome to **BriCANet**, the companion repository for "Computer Vision-based seismic assessment of RC simply supported bridges characterized by corroded circular piers".

This repository provides you with two scripts:

* `BriCANET_training.py`, which can be used to train the BriCANET model on your data.
* `BriCANET_GUI.py`, which is a powerful GUI application for corrosion intensity classification in RC structures images using the *BriCANET*.

This guide will help you create a standalone executable from the Python source code.

📸 *BriCANET Framework*  
![folder-structure](images/BriCANET_framework.png)  

---

## 🚀 Quick Start - Create Your Executable

### :alarm_clock: 0. Prepare your data

The BriCANet model is expected to have a dataset composed by three different classes, that is, *low*, *medium*, and *high*, each one referring to a different level of corrosion. Therefore, the data folder should be arranged according to the following structure.

```bash
📁 dataset_path
├── 📁 low              (Images showing low corrosion)
├── 📁 medium           (Images showing medium corrosion)
└── 📁 high             (Images showing high corrosion)
```

### :chart: 1. Train the model

Run the `BriCANet_training.py` script as follows.

```sh
python BriCANet_training.py -d dataset_path
```

Where `dataset_path` is the folder arranged at step 0.

### 📂 2. Prepare the files

1. Create a folder on your computer (example: `C:\BriCANet\`)  
2. Save the file **BriCANet_GUI.py** inside this folder  
3. Save the file **requirements.txt** inside this folder  


### ⚙️ 3. Install everything you need
⚠️ **Important:** Make sure you have **Python 3.12.2** installed on your system before continuing.  
You can download it from the official website: [**Python**](https://www.python.org/downloads/).  
When installing, check the option **“Add Python to PATH”**.  


1. Open the Command Prompt (CMD): type `cmd` and press **Enter**  
   ![open-cmd](images/open-cmd.gif)  

2. Navigate to the project folder:
   ```bash
   cd C:\BriCANet
   ```

3. Create a virtual environment:
   ```bash
   python -m venv BriCANET_env
   ```

4. Activate the virtual environmen:
   - On Windows:
   ```bash
   BriCANET_env\Scripts\activate
   ```
   - On macOS/Linux:
   ```bash
   source BriCANET_env/bin/activate
   ```

5. Upgrade pip to the latest version:
   ```bash
   pip install --upgrade pip
   ```

6. Install the required Python libraries with: 
   ```bash
   pip install -r requirements.txt
   ```

7. Install PyInstaller (to create the executable) with:
   ```bash
   pip install pyinstaller
   ```


### 🛠️ 4. Create the executable 

Run this command in the terminal:

   ```bash
   pyinstaller --onefile --windowed --name="BriCANet_Classifier" --clean --noconsole --hidden-import=tensorflow --hidden-import=pil BriCANet_GUI.py
   ```

Wait until PyInstaller finishes and...done! BriCANet is ready to predict!  :fire:

---

# 📊 Expected Folder Structure

```bash
C:\BriCANet\
├── 📄 BriCANet_GUI.py          (Main application code)
├── 📄 requirements.txt         (Dependencies list)
├── 📁 dist/
│   └── 🎯 BriCANet_Classifier.exe  (#Your final executable!)
└── 📁 build/                   (Temporary build files)
```

## 🚀 How to Use the App

1. Launch the application by running the Python file.  
2. Load the `.h5` model using the dedicated button.  
   - You can create this model by training it with the **`BriCANET_training.py`** script on your chosen dataset.  
3. For each face:  
   - Select **"Corrosion Detected"** if corrosion is present.  
   - Upload the corresponding image.  
4. Click **"Run Full Analysis"** to generate all results.

### 🎥 Demo Video

![folder-structure](images/Tutorial.gif)

---


## ⚠️ Important Disclaimer & Development Status

![folder-structure](images/Disclaimer.gif)

### 🔬 Research Development Phase

BriCANet is currently in active development and improvement phase, so this tool should be considered as a research prototype.

### 📚 Cite this work

If you use BriCANet, please cite the following work.

```
@article{XXXXXXXXXXXXX,
  title={XXXXXXXXXXXXXXXXXXX},
  author={XXXXXXXXXXXXXXX},
  journal={XXXXXXXXXXXXXXXX},
  year={2025},
  publisher={XXXXXXXXXXXXXXXXXXXX}
}
```

### 🏆 Institutions & Collaborators

This repository is maintained by this wonderful team.


| Wonderful guy | Affiliation | How to reach | Contacts |
| ------------- | ----------- | ------------ | -------- |
| Vincenzo Mario Di Mucci | POLIBA | [v.dimucci1@phd.poliba.it](mailto:v.dimucci1@phd.poliba.it) | [![Scopus](https://img.shields.io/badge/Scopus-Profile-orange?logo=Elsevier&logoColor=white)](https://www.scopus.com/authid/detail.uri?authorId=59278301400) [![ResearchGate](https://img.shields.io/badge/ResearchGate-Profile-00CCBB?logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Vincenzo-Di-Mucci) |
| Angelo Cardellicchio | STIIMA | [angelo.cardellicchio@cnr.it](mailto:angelo.cardellicchio@cnr.it) | [![Scopus](https://img.shields.io/badge/Scopus-Profile-orange?logo=Elsevier&logoColor=white)](https://www.scopus.com/authid/detail.uri?authorId=56786372800) [![ResearchGate](https://img.shields.io/badge/ResearchGate-Profile-00CCBB?logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Angelo-Cardellicchio) |
| Sergio Ruggieri | POLIBA | [sergio.ruggieri@poliba.it](mailto:sergio.ruggieri@poliba.it) | [![Scopus](https://img.shields.io/badge/Scopus-Profile-orange?logo=Elsevier&logoColor=white)](https://www.scopus.com/authid/detail.uri?authorId=57200721168) [![ResearchGate](https://img.shields.io/badge/ResearchGate-Profile-00CCBB?logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Sergio-Ruggieri-2) |
| Andrea Nettis | POLIBA | [andrea.nettis@poliba.it](mailto:andrea.nettis@poliba.it) | [![Scopus](https://img.shields.io/badge/Scopus-Profile-orange?logo=Elsevier&logoColor=white)](https://www.scopus.com/authid/detail.uri?authorId=57214778072) [![ResearchGate](https://img.shields.io/badge/ResearchGate-Profile-00CCBB?logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Andrea-Nettis-2) |
| Vito Renò | STIIMA | [vito.reno@cnr.it](mailto:vito.reno@cnr.it) | [![Scopus](https://img.shields.io/badge/Scopus-Profile-orange?logo=Elsevier&logoColor=white)](https://www.scopus.com/authid/detail.uri?authorId=56433738300&source=sd-apx&adobe_mc=MCMID%3D08848743994237898301143928221677623271%7CMCORGID%3D4D6368F454EC41940A4C98A6%2540AdobeOrg%7CTS%3D1759477479) [![ResearchGate](https://img.shields.io/badge/ResearchGate-Profile-00CCBB?logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Vito-Reno) |
| Giuseppina Uva | POLIBA | [giuseppina.uva@poliba.it](mailto:giuseppina.uva@poliba.it) | [![Scopus](https://img.shields.io/badge/Scopus-Profile-orange?logo=Elsevier&logoColor=white)](https://www.scopus.com/authid/detail.uri?authorId=12143743700&source=sd-apx&adobe_mc=MCMID%3D08848743994237898301143928221677623271%7CMCORGID%3D4D6368F454EC41940A4C98A6%2540AdobeOrg%7CTS%3D1759477503) [![ResearchGate](https://img.shields.io/badge/ResearchGate-Profile-00CCBB?logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Giuseppina-Uva) |

---
<table>
  <tr>
    <td>
      <img src="images/logos/poliba.png" alt="Poliba Logo" width="250"/>
    </td>
    <td>
      <strong>POLIBA</strong> – Politecnico di Bari, DICATECh, Via Orabona 4, Bari, 70126, Italy
    </td>
  </tr>
  <tr>
    <td>
      <img src="images/logos/stiima.png" alt="CNR Logo" width="100"/>
    </td>
    <td>
      <strong>STIIMA</strong> – Istituto di Sistemi e Tecnologie Industriali Intelligenti per il Manifatturiero Avanzato, Consiglio Nazionale delle Ricerche, Via Amendola 122 D/O, Bari, 70126, Italy
    </td>
  </tr>
</table>


