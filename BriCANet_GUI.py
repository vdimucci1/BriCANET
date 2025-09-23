import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import os
import threading

class CorrosionClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificatore Corrosione Pila dalle immagini - BriCANet")
        self.root.geometry("800x800")
        self.root.configure(bg="#f0f0f0")
        
        # Variabili di stato
        self.model = None
        self.class_names = ['Alto', 'Basso', 'Medio']
        self.face_images = {
            'Faccia 1': None,
            'Faccia 2': None,
            'Faccia 3': None,
            'Faccia 4': None
        }
        self.face_image_paths = {
            'Faccia 1': None,
            'Faccia 2': None,
            'Faccia 3': None,
            'Faccia 4': None
        }
        
        self.create_widgets()
    
    def create_widgets(self):
        # Tema moderno
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f8f9fa")
        style.configure("TLabelFrame", background="#f8f9fa", font=("Segoe UI", 11, "bold"))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.map("TButton",
                foreground=[("active", "#ffffff")],
                background=[("active", "#3498db")])

        # Titolo principale
        title_frame = tk.Frame(self.root, bg="#2c3e50")
        title_frame.pack(fill="x", pady=(0, 10))

        title_label = tk.Label(
            title_frame,
            text="⚡ BriCANet - Classificatore Corrosione Pile da ponte",
            font=("Segoe UI", 22, "bold"),
            fg="white",
            bg="#2c3e50",
            pady=20
        )
        title_label.pack()

        # Canvas con scrollbar
        main_canvas = tk.Canvas(self.root, bg="#f8f9fa", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )

        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        # Frame caricamento modello
        model_frame = ttk.LabelFrame(scrollable_frame, text="Caricamento Modello", padding=12)
        model_frame.pack(fill="x", padx=15, pady=8)

        ttk.Button(
            model_frame,
            text="📂 Carica Modello BriCANet (.h5)",
            command=self.load_model
        ).pack(side="left", padx=5)

        self.model_status = tk.Label(
            model_frame,
            text="Nessun modello caricato",
            fg="red",
            bg="#f8f9fa",
            font=("Segoe UI", 10, "italic")
        )
        self.model_status.pack(side="left", padx=10)

        # Frame immagini
        images_frame = ttk.LabelFrame(scrollable_frame, text="Immagini delle 4 Facce", padding=12)
        images_frame.pack(fill="both", expand=True, padx=15, pady=8)

        faces_container = tk.Frame(images_frame, bg="#f8f9fa")
        faces_container.pack(fill="both", expand=True)

        self.face_frames, self.face_labels, self.face_buttons = {}, {}, {}

        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i, (face_name, (row, col)) in enumerate(zip(self.face_images.keys(), positions)):
            face_frame = tk.Frame(faces_container, bg="white", relief="groove", bd=2)
            face_frame.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")

            faces_container.grid_rowconfigure(row, weight=1)
            faces_container.grid_columnconfigure(col, weight=1)

            title_label = tk.Label(
                face_frame,
                text=face_name,
                font=("Segoe UI", 13, "bold"),
                bg="white",
                fg="#2c3e50"
            )
            title_label.pack(pady=5)

            image_label = tk.Label(
                face_frame,
                text="🖼️ Nessuna immagine\ncaricata",
                bg="#eaeaea",
                width=25,
                height=12,
                relief="sunken",
                font=("Segoe UI", 10)
            )
            image_label.pack(padx=10, pady=5)

            load_button = ttk.Button(
                face_frame,
                text=f"📁 Carica {face_name}",
                command=lambda fn=face_name: self.load_image(fn)
            )
            load_button.pack(pady=5)

            self.face_frames[face_name] = face_frame
            self.face_labels[face_name] = image_label
            self.face_buttons[face_name] = load_button

        # Frame controlli
        controls_frame = ttk.LabelFrame(scrollable_frame, text="Controlli", padding=12)
        controls_frame.pack(fill="x", padx=15, pady=8)

        buttons_frame = tk.Frame(controls_frame, bg="#f8f9fa")
        buttons_frame.pack(fill="x")

        self.predict_button = ttk.Button(
            buttons_frame,
            text="🔍 Avvia Classificazione",
            command=self.start_prediction,
            state="disabled"
        )
        self.predict_button.pack(side="left", padx=5)

        ttk.Button(
            buttons_frame,
            text="🗑️ Pulisci Tutto",
            command=self.clear_all
        ).pack(side="left", padx=5)

        self.progress = ttk.Progressbar(controls_frame, mode='indeterminate')
        self.progress.pack(fill="x", pady=8)

        # Frame risultati
        results_frame = ttk.LabelFrame(scrollable_frame, text="Risultati Classificazione", padding=12)
        results_frame.pack(fill="both", expand=True, padx=15, pady=8)

        self.results_text = tk.Text(
            results_frame,
            height=15,
            font=("Consolas", 11),
            wrap="word",
            bg="#1e272e",
            fg="#d2dae2",
            insertbackground="white"
        )
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)

        self.results_text.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")

        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)

    
    def load_model(self):
        """Carica il modello MCCA-Net"""
        file_path = filedialog.askopenfilename(
            title="Seleziona il modello MCCA-Net",
            filetypes=[("Modelli H5", "*.h5"), ("Tutti i file", "*.*")]
        )
        
        if file_path:
            try:
                self.model = load_model(file_path)
                self.model_status.config(text=f"Modello caricato: {os.path.basename(file_path)}", fg="green")
                self.update_predict_button_state()
                self.log_result(f"✅ Modello caricato con successo: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Errore", f"Errore nel caricamento del modello:\n{str(e)}")
                self.log_result(f"❌ Errore nel caricamento del modello: {str(e)}")
    
    def load_image(self, face_name):
        """Carica un'immagine per una faccia specifica"""
        file_path = filedialog.askopenfilename(
            title=f"Seleziona immagine per {face_name}",
            filetypes=[
                ("Immagini", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Tutti i file", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Carica e ridimensiona l'immagine per la visualizzazione
                pil_image = Image.open(file_path)
                pil_image.thumbnail((200, 150), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Aggiorna la label con l'immagine
                self.face_labels[face_name].config(image=photo, text="")
                self.face_labels[face_name].image = photo  # Mantieni il riferimento
                
                # Salva il percorso dell'immagine
                self.face_image_paths[face_name] = file_path
                self.face_images[face_name] = file_path
                
                self.update_predict_button_state()
                self.log_result(f"📁 Immagine caricata per {face_name}: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Errore", f"Errore nel caricamento dell'immagine:\n{str(e)}")
                self.log_result(f"❌ Errore nel caricamento immagine {face_name}: {str(e)}")
    
    def update_predict_button_state(self):
        """Aggiorna lo stato del bottone di predizione"""
        if self.model and all(img is not None for img in self.face_images.values()):
            self.predict_button.config(state="normal")
        else:
            self.predict_button.config(state="disabled")
    
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
        """Genera patch usando sliding window"""
        patches = []
        h, w, _ = image.shape
        for y in range(0, h - window_size[1] + 1, step_size):
            for x in range(0, w - window_size[0] + 1, step_size):
                patch = image[y:y + window_size[1], x:x + window_size[0]]
                patches.append(patch)
        return patches
    
    def preprocess_new_image(self, image_path):
        """Preprocessa una nuova immagine"""
        # Carica l'immagine e convertila in RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Applica la fusione degli spazi colore
        fused_image = self.fuse_color_spaces(image)
        
        # Dividi l'immagine in patch con lo sliding window
        window_size = (224, 224)
        step_size = 112
        patches = self.sliding_window(fused_image, window_size, step_size)
        
        # Normalizza le patch
        processed_patches = [patch / 255.0 for patch in patches]
        
        return np.array(processed_patches)
    
    def predict_single_image(self, image_path, face_name):
        """Effettua la predizione su una singola immagine"""
        try:
            # Preprocessa l'immagine
            patches = self.preprocess_new_image(image_path)
            
            # Predici su ogni patch
            predictions = self.model.predict(patches, verbose=0)
            avg_prediction = np.mean(predictions, axis=0)
            
            # Ottieni la classe predetta e le probabilità
            predicted_class_idx = np.argmax(avg_prediction)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = avg_prediction[predicted_class_idx]
            
            return {
                'face': face_name,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': avg_prediction,
                'num_patches': len(patches)
            }
            
        except Exception as e:
            return {
                'face': face_name,
                'error': str(e)
            }
    
    def start_prediction(self):
        """Avvia la predizione in un thread separato"""
        threading.Thread(target=self.run_prediction, daemon=True).start()
    
    def run_prediction(self):
        """Esegue la predizione su tutte le immagini"""
        try:
            # Mostra la progress bar
            self.progress.start(10)
            self.predict_button.config(state="disabled")
            
            self.log_result("🚀 Iniziando la classificazione delle 4 facce...")
            self.log_result("="*60)
            
            results = []
            
            for face_name, image_path in self.face_image_paths.items():
                if image_path:
                    self.log_result(f"🔍 Elaborando {face_name}...")
                    result = self.predict_single_image(image_path, face_name)
                    results.append(result)
                    
                    if 'error' in result:
                        self.log_result(f"❌ Errore in {face_name}: {result['error']}")
                    else:
                        self.log_result(f"✅ {face_name} completata")
            
            # Mostra i risultati dettagliati
            self.display_results(results)
            
        except Exception as e:
            self.log_result(f"❌ Errore generale: {str(e)}")
            messagebox.showerror("Errore", f"Errore durante la predizione:\n{str(e)}")
        
        finally:
            # Ferma la progress bar e riabilita il bottone
            self.progress.stop()
            self.predict_button.config(state="normal")
    
    def display_results(self, results):
        """Visualizza i risultati dettagliati"""
        self.log_result("\n📊 RISULTATI CLASSIFICAZIONE")
        self.log_result("="*60)
        
        summary = {}
        
        for result in results:
            if 'error' not in result:
                face = result['face']
                predicted_class = result['predicted_class']
                confidence = result['confidence']
                probabilities = result['probabilities']
                num_patches = result['num_patches']
                
                self.log_result(f"\n🔸 {face}:")
                self.log_result(f"   Classe predetta: {predicted_class}")
                self.log_result(f"   Confidenza: {confidence:.4f} ({confidence*100:.2f}%)")
                self.log_result(f"   Numero di patch analizzate: {num_patches}")
                self.log_result(f"   Vettore probabilità:")
                
                for i, class_name in enumerate(self.class_names):
                    prob = probabilities[i]
                    bar = "█" * int(prob * 20)  # Barra visuale
                    self.log_result(f"     {class_name:10}: {prob:.4f} |{bar:<20}| {prob*100:.2f}%")
                
                # Aggiorna il sommario
                if predicted_class not in summary:
                    summary[predicted_class] = 0
                summary[predicted_class] += 1
        
        # Mostra il sommario generale
        self.log_result("\n📈 SOMMARIO GENERALE:")
        self.log_result("="*30)
        total_faces = len([r for r in results if 'error' not in r])
        
        for class_name in ['Alto', 'Medio', 'Basso']:
            count = summary.get(class_name, 0)
            percentage = (count / total_faces * 100) if total_faces > 0 else 0
            self.log_result(f"{class_name:10}: {count}/{total_faces} facce ({percentage:.1f}%)")
        
        if summary:
            most_common = max(summary, key=summary.get)
            self.log_result(f"\n🎯 Classe predominante: {most_common}")
        
        self.log_result("\n" + "="*60)
        self.log_result("✅ Classificazione completata!")
    
    def log_result(self, message):
        """Aggiunge un messaggio ai risultati"""
        def update_text():
            self.results_text.insert(tk.END, message + "\n")
            self.results_text.see(tk.END)
        
        self.root.after(0, update_text)
    
    def clear_all(self):
        """Pulisce tutti i dati"""
        # Reset immagini
        for face_name in self.face_images.keys():
            self.face_images[face_name] = None
            self.face_image_paths[face_name] = None
            self.face_labels[face_name].config(
                image="",
                text="Nessuna immagine\ncaricata"
            )
            self.face_labels[face_name].image = None
        
        # Reset risultati
        self.results_text.delete(1.0, tk.END)
        
        # Reset stato bottone
        self.update_predict_button_state()
        
        self.log_result("🧹 Tutti i dati sono stati puliti.")

def main():
    root = tk.Tk()
    app = CorrosionClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()