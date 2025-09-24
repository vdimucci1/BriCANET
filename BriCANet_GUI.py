import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import os
from pathlib import Path

class CorrosionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Corrosion Detection & Classification System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a2e')
        
        # Variabili dell'applicazione
        self.model = None
        self.class_names = ['Alto', 'Basso', 'Medio']
        self.faces_data = {f'face_{i+1}': {'has_corrosion': tk.BooleanVar(), 'image_path': None, 'result': None} for i in range(4)}
        
        # Configurazione degli stili
        self.setup_styles()
        
        # Creazione dell'interfaccia
        self.create_widgets()
        
    def setup_styles(self):
        """Configura gli stili per un'interfaccia moderna"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colori principali
        bg_color = '#1a1a2e'
        secondary_bg = '#16213e'
        accent_color = '#0f4c75'
        text_color = '#ffffff'
        button_color = '#3282b8'
        success_color = '#00ff41'
        warning_color = '#ff6b35'
        
        # Configurazione degli stili
        style.configure('Title.TLabel', 
                       background=bg_color, 
                       foreground=text_color, 
                       font=('Arial', 24, 'bold'))
        
        style.configure('Heading.TLabel', 
                       background=bg_color, 
                       foreground=text_color, 
                       font=('Arial', 14, 'bold'))
        
        style.configure('Modern.TFrame', 
                       background=secondary_bg, 
                       relief='flat', 
                       borderwidth=2)
        
        style.configure('Card.TFrame', 
                       background=accent_color, 
                       relief='raised', 
                       borderwidth=3)
        
        style.configure('Modern.TButton', 
                       background=button_color, 
                       foreground=text_color, 
                       font=('Arial', 10, 'bold'),
                       borderwidth=0)
        
        style.map('Modern.TButton',
                 background=[('active', '#4a9fd1'), ('pressed', '#2c5282')])
        
        style.configure('Success.TLabel', 
                       background=accent_color, 
                       foreground=success_color, 
                       font=('Arial', 11, 'bold'))
        
        style.configure('Warning.TLabel', 
                       background=accent_color, 
                       foreground=warning_color, 
                       font=('Arial', 11, 'bold'))
        
    def create_widgets(self):
        """Crea tutti i widget dell'interfaccia"""
        # Frame principale
        main_frame = ttk.Frame(self.root, style='Modern.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Titolo
        title_label = ttk.Label(main_frame, 
                               text="🔍 Corrosion Detection & Classification System", 
                               style='Title.TLabel')
        title_label.pack(pady=(0, 30))
        
        # Frame per caricamento modello
        model_frame = ttk.Frame(main_frame, style='Card.TFrame')
        model_frame.pack(fill=tk.X, pady=(0, 20), padx=10, ipady=15)
        
        model_label = ttk.Label(model_frame, 
                               text="🤖 Carica Modello H5", 
                               style='Heading.TLabel')
        model_label.pack(pady=(10, 5))
        
        model_button = ttk.Button(model_frame, 
                                 text="Seleziona Modello (.h5)", 
                                 command=self.load_model,
                                 style='Modern.TButton')
        model_button.pack(pady=(0, 10))
        
        self.model_status_label = ttk.Label(model_frame, 
                                           text="❌ Nessun modello caricato", 
                                           style='Warning.TLabel')
        self.model_status_label.pack()
        
        # Frame per le facce
        faces_frame = ttk.Frame(main_frame, style='Modern.TFrame')
        faces_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Griglia 2x2 per le 4 facce
        for i in range(4):
            row = i // 2
            col = i % 2
            self.create_face_widget(faces_frame, i + 1, row, col)
        
        # Frame per i risultati
        results_frame = ttk.Frame(main_frame, style='Card.TFrame')
        results_frame.pack(fill=tk.X, pady=20, padx=10, ipady=15)
        
        results_label = ttk.Label(results_frame, 
                                 text="📊 Analizza Tutte le Facce", 
                                 style='Heading.TLabel')
        results_label.pack(pady=(10, 10))
        
        analyze_button = ttk.Button(results_frame, 
                                   text="🚀 Avvia Analisi Completa", 
                                   command=self.analyze_all_faces,
                                   style='Modern.TButton')
        analyze_button.pack(pady=(0, 10))
        
        # Area risultati scrollabile
        self.create_results_area(results_frame)
    
    def create_face_widget(self, parent, face_num, row, col):
        """Crea il widget per una singola faccia"""
        face_key = f'face_{face_num}'
        
        # Frame principale della faccia
        face_frame = ttk.Frame(parent, style='Card.TFrame')
        face_frame.grid(row=row, column=col, padx=15, pady=15, sticky='nsew', ipadx=20, ipady=20)
        parent.grid_columnconfigure(col, weight=1)
        parent.grid_rowconfigure(row, weight=1)
        
        # Titolo della faccia
        title = ttk.Label(face_frame, 
                         text=f"🔧 Faccia {face_num}", 
                         style='Heading.TLabel')
        title.pack(pady=(0, 15))
        
        # Checkbox per corrosione rilevata
        corrosion_check = ttk.Checkbutton(face_frame, 
                                         text="Corrosione Rilevata", 
                                         variable=self.faces_data[face_key]['has_corrosion'],
                                         command=lambda: self.toggle_image_upload(face_key))
        corrosion_check.pack(pady=(0, 10))
        
        # Frame per upload immagine (inizialmente nascosto)
        upload_frame = ttk.Frame(face_frame, style='Modern.TFrame')
        upload_frame.pack(fill=tk.X, pady=10)
        
        # Bottone per caricare immagine
        upload_button = ttk.Button(upload_frame, 
                                  text="📁 Carica Immagine", 
                                  command=lambda: self.load_image(face_key),
                                  style='Modern.TButton',
                                  state='disabled')
        upload_button.pack(pady=5)
        
        # Label per mostrare il nome del file
        file_label = ttk.Label(upload_frame, 
                              text="Nessun file selezionato", 
                              background='#16213e',
                              foreground='#888888',
                              font=('Arial', 9))
        file_label.pack(pady=5)
        
        # Area per i risultati
        result_frame = ttk.Frame(face_frame, style='Modern.TFrame')
        result_frame.pack(fill=tk.X, pady=(10, 0))
        
        result_label = ttk.Label(result_frame, 
                                text="", 
                                background='#16213e',
                                foreground='#ffffff',
                                font=('Arial', 10),
                                wraplength=250)
        result_label.pack()
        
        # Salva riferimenti ai widget
        setattr(self, f'{face_key}_upload_frame', upload_frame)
        setattr(self, f'{face_key}_upload_button', upload_button)
        setattr(self, f'{face_key}_file_label', file_label)
        setattr(self, f'{face_key}_result_label', result_label)
    
    def create_results_area(self, parent):
        """Crea l'area scrollabile per i risultati"""
        # Frame con scrollbar
        canvas = tk.Canvas(parent, bg='#0f4c75', height=200)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas, style='Modern.TFrame')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        # Label iniziale
        initial_label = ttk.Label(self.scrollable_frame,
                                 text="I risultati dell'analisi appariranno qui...",
                                 background='#16213e',
                                 foreground='#888888',
                                 font=('Arial', 11))
        initial_label.pack(pady=20)
    
    def load_model(self):
        """Carica il modello H5"""
        file_path = filedialog.askopenfilename(
            title="Seleziona il file del modello",
            filetypes=[("File H5", "*.h5"), ("Tutti i file", "*.*")]
        )
        
        if file_path:
            try:
                self.model = load_model(file_path)
                self.model_status_label.config(
                    text=f"✅ Modello caricato: {Path(file_path).name}",
                    style='Success.TLabel'
                )
                messagebox.showinfo("Successo", "Modello caricato correttamente!")
            except Exception as e:
                messagebox.showerror("Errore", f"Impossibile caricare il modello: {str(e)}")
                self.model_status_label.config(
                    text="❌ Errore nel caricamento del modello",
                    style='Warning.TLabel'
                )
    
    def toggle_image_upload(self, face_key):
        """Abilita/disabilita l'upload dell'immagine"""
        upload_button = getattr(self, f'{face_key}_upload_button')
        
        if self.faces_data[face_key]['has_corrosion'].get():
            upload_button.config(state='normal')
        else:
            upload_button.config(state='disabled')
            # Reset dei dati se la corrosione viene deselezionata
            self.faces_data[face_key]['image_path'] = None
            self.faces_data[face_key]['result'] = None
            file_label = getattr(self, f'{face_key}_file_label')
            file_label.config(text="Nessun file selezionato")
            result_label = getattr(self, f'{face_key}_result_label')
            result_label.config(text="")
    
    def load_image(self, face_key):
        """Carica un'immagine per una faccia specifica"""
        file_path = filedialog.askopenfilename(
            title=f"Seleziona immagine per {face_key.replace('_', ' ').title()}",
            filetypes=[
                ("Immagini", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Tutti i file", "*.*")
            ]
        )
        
        if file_path:
            self.faces_data[face_key]['image_path'] = file_path
            file_label = getattr(self, f'{face_key}_file_label')
            file_label.config(text=f"📷 {Path(file_path).name}")
    
    def fuse_color_spaces(self, image):
        """Fusione degli spazi colore"""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
        h, s, v = cv2.split(hsv_image)
        l, a, b = cv2.split(lab_image)
        fused_image = cv2.merge((h + l, s + a, v + b))
        return fused_image
    
    def sliding_window(self, image, window_size, step_size):
        """Sliding window per dividere l'immagine in patch"""
        patches = []
        h, w, _ = image.shape
        for y in range(0, h - window_size[1] + 1, step_size):
            for x in range(0, w - window_size[0] + 1, step_size):
                patch = image[y:y + window_size[1], x:x + window_size[0]]
                patches.append(patch)
        return patches
    
    def preprocess_image(self, image_path):
        """Preprocessa un'immagine per la predizione"""
        # Carica l'immagine
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Applica la fusione degli spazi colore
        fused_image = self.fuse_color_spaces(image)
        
        # Dividi in patch
        window_size = (224, 224)
        step_size = 112
        patches = self.sliding_window(fused_image, window_size, step_size)
        
        # Normalizza le patch
        processed_patches = [patch / 255.0 for patch in patches]
        
        return np.array(processed_patches)
    
    def predict_single_image(self, image_path):
        """Effettua la predizione su una singola immagine"""
        if not self.model:
            raise ValueError("Modello non caricato")
        
        # Preprocessa l'immagine
        patches = self.preprocess_image(image_path)
        
        # Predici su ogni patch
        predictions = self.model.predict(patches, verbose=0)
        avg_prediction = np.mean(predictions, axis=0)
        
        # Ottieni la classe predetta e le probabilità
        predicted_class_idx = np.argmax(avg_prediction)
        predicted_class = self.class_names[predicted_class_idx]
        probabilities = avg_prediction
        
        return predicted_class, probabilities
    
    def analyze_all_faces(self):
        """Analizza tutte le facce selezionate"""
        if not self.model:
            messagebox.showerror("Errore", "Carica prima un modello!")
            return
        
        # Pulisci l'area risultati
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Header risultati
        header = ttk.Label(self.scrollable_frame,
                          text="🔬 RISULTATI ANALISI CORROSIONE",
                          style='Heading.TLabel')
        header.pack(pady=(20, 30))
        
        results_found = False
        
        # Analizza ogni faccia
        for i in range(1, 5):
            face_key = f'face_{i}'
            face_data = self.faces_data[face_key]
            
            if face_data['has_corrosion'].get():
                results_found = True
                
                # Frame per questa faccia
                face_result_frame = ttk.Frame(self.scrollable_frame, style='Card.TFrame')
                face_result_frame.pack(fill=tk.X, pady=10, padx=20, ipady=15)
                
                face_title = ttk.Label(face_result_frame,
                                      text=f"🔧 FACCIA {i}",
                                      style='Heading.TLabel')
                face_title.pack(pady=(10, 15))
                
                if face_data['image_path']:
                    try:
                        # Effettua la predizione
                        predicted_class, probabilities = self.predict_single_image(face_data['image_path'])
                        
                        # Mostra i risultati
                        class_label = ttk.Label(face_result_frame,
                                              text=f"🎯 Classe Predetta: {predicted_class}",
                                              background='#0f4c75',
                                              foreground='#00ff41',
                                              font=('Arial', 12, 'bold'))
                        class_label.pack(pady=5)
                        
                        # Vettore probabilità
                        prob_text = "📊 Vettore Probabilità:\n"
                        for j, class_name in enumerate(self.class_names):
                            prob_text += f"   {class_name}: {probabilities[j]:.4f} ({probabilities[j]*100:.2f}%)\n"
                        
                        prob_label = ttk.Label(face_result_frame,
                                             text=prob_text,
                                             background='#0f4c75',
                                             foreground='#ffffff',
                                             font=('Arial', 10),
                                             justify=tk.LEFT)
                        prob_label.pack(pady=10)
                        
                        # Salva i risultati
                        self.faces_data[face_key]['result'] = {
                            'predicted_class': predicted_class,
                            'probabilities': probabilities
                        }
                        
                        # Aggiorna anche il widget della faccia
                        result_label = getattr(self, f'{face_key}_result_label')
                        result_text = f"🎯 {predicted_class}\n📊 Conf: {max(probabilities)*100:.1f}%"
                        result_label.config(text=result_text, foreground='#00ff41')
                        
                    except Exception as e:
                        error_label = ttk.Label(face_result_frame,
                                              text=f"❌ Errore nell'analisi: {str(e)}",
                                              background='#0f4c75',
                                              foreground='#ff6b35',
                                              font=('Arial', 10))
                        error_label.pack(pady=10)
                        
                        # Aggiorna il widget della faccia con errore
                        result_label = getattr(self, f'{face_key}_result_label')
                        result_label.config(text="❌ Errore", foreground='#ff6b35')
                else:
                    warning_label = ttk.Label(face_result_frame,
                                            text="⚠️ Nessuna immagine caricata",
                                            background='#0f4c75',
                                            foreground='#ff6b35',
                                            font=('Arial', 11))
                    warning_label.pack(pady=10)
        
        if not results_found:
            no_results_label = ttk.Label(self.scrollable_frame,
                                       text="ℹ️ Nessuna faccia con corrosione selezionata",
                                       background='#16213e',
                                       foreground='#888888',
                                       font=('Arial', 12))
            no_results_label.pack(pady=50)

def main():
    """Funzione principale per avviare l'applicazione"""
    root = tk.Tk()
    app = CorrosionDetectionApp(root)
    
    # Centra la finestra sullo schermo
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Icona della finestra (opzionale)
    try:
        root.iconbitmap('icon.ico')  # Aggiungi un'icona se disponibile
    except:
        pass
    
    root.mainloop()

if __name__ == "__main__":
    main()
