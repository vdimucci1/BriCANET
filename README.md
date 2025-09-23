# BriCANet Classifier - GUI Application 🧠

Welcome to **BriCANet**, a powerful GUI application for corrosion intensity classification in RC structures images using deep learning. 
This guide will help you create a standalone executable from the Python source code.

---

## 🚀 Quick Start - Create Your Executable

### 📂 1. Prepare the files

1. Create a folder on your computer (example: `C:\BriCANet\`)  
2. Save the file **BriCANet_GUI.py** inside this folder  
3. Save the file **requirements.txt** inside this folder  

📸 *Example folder screenshot here*  
![folder-structure](images/folder.png)  


### ⚙️ 2. Install everything you need

1. Open the Command Prompt (CMD): type `cmd` and press **Enter**  
   ![open-cmd](images/open-cmd.gif)  

2. Navigate to the project folder:
   ```bash
   cd C:\BriCANet

3. Install the required Python libraries with: 
   ```bash
   pip install -r requirements.txt

4. Install PyInstaller (to create the executable) with:
   ```bash
   pip install pyinstaller


### 🛠️ 3. Create the executable

Run this command in the terminal:
   ```bash
   pyinstaller --onefile --windowed --name="BriCANet_Classifier" --clean --noconsole --hidden-import=tensorflow --hidden-import=pil BriCANet_GUI.py

