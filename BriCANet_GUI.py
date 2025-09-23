import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk, ImageDraw 
import os
import threading
import sys


class CorrosionClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ BriCANet - Classificatore Avanzato Corrosione Pile")
        self.root.geometry("900x900")
        self.root.configure(bg="#1e1e2e")
        self.root.minsize(850, 800)
        
        # Centrare la finestra
        self.center_window()
        
        # Variabili di stato
        self.model = None
        self.class_names = ['Alto', 'Medio', 'Basso']
        self.class_colors = {
            'Alto': '#ff6b6b',
            'Medio': '#ffd93d',
            'Basso': '#6bcf7f'
        }
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
    
    def center_window(self):
        """Centra la finestra sullo schermo"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        # Tema moderno e scuro
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurazione stili personalizzati
        style.configure('Custom.TFrame', background='#2d2d44')
        style.configure('Custom.TLabelframe', background='#2d2d44', foreground='white')
        style.configure('Custom.TLabelframe.Label', background='#2d2d44', foreground='white')
        style.configure('Custom.TButton', 
                       background='#4ecdc4', 
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none')
        style.map('Custom.TButton',
                 background=[('active', '#45b7af'), ('pressed', '#3ca19a')])
        
        style.configure('Red.TButton', 
                       background='#ff6b6b', 
                       foreground='white')
        style.map('Red.TButton',
                 background=[('active', '#e55a5a'), ('pressed', '#cc4f4f')])
        
        style.configure('Green.TButton', 
                       background='#6bcf7f', 
                       foreground='white')
        style.map('Green.TButton',
                 background=[('active', '#5dbd70'), ('pressed', '#4fa860')])

        # Header con gradiente
        header_frame = tk.Frame(self.root, bg='#1e1e2e', height=120)
        header_frame.pack(fill='x', pady=(0, 10))
        header_frame.pack_propagate(False)

        # Titolo principale con effetto moderno
        title_label = tk.Label(
            header_frame,
            text="⚡ BriCANet",
            font=("Segoe UI", 28, "bold"),
            fg="#4ecdc4",
            bg="#1e1e2e"
        )
        title_label.pack(pady=(20, 5))

        subtitle_label = tk.Label(
            header_frame,
            text="Sistema Intelligente di Classificazione Corrosione Pile da Ponte",
            font=("Segoe UI", 12),
            fg="#a5b3c1",
            bg="#1e1e2e"
        )
        subtitle_label.pack()

        # Container principale con scrollbar
        main_container = tk.Frame(self.root, bg="#1e1e2e")
        main_container.pack(fill='both', expand=True, padx=15, pady=5)

        # Canvas con scrollbar
        canvas = tk.Canvas(main_container, bg="#1e1e2e", highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style='Custom.TFrame')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Frame caricamento modello
        model_frame = ttk.LabelFrame(
            scrollable_frame, 
            text="🎯 CARICAMENTO MODELLO DI INTELLIGENZA ARTIFICIALE", 
            style='Custom.TLabelframe',
            padding=15
        )
        model_frame.pack(fill='x', pady=10, ipady=5)

        model_inner_frame = tk.Frame(model_frame, bg="#2d2d44")
        model_inner_frame.pack(fill='x')

        ttk.Button(
            model_inner_frame,
            text="📁 Seleziona Modello BriCANet (.h5)",
            style='Custom.TButton',
            command=self.load_model
        ).pack(side='left', padx=5, pady=10)

        self.model_status = tk.Label(
            model_inner_frame,
            text="⏳ Nessun modello caricato",
            fg="#ffd93d",
            bg="#2d2d44",
            font=("Segoe UI", 10, "bold")
        )
        self.model_status.pack(side='left', padx=15, pady=10)

        # Frame immagini con layout a griglia
        images_frame = ttk.LabelFrame(
            scrollable_frame, 
            text="📸 GALLERIA IMMAGINI - 4 FACCE DELLA PILA", 
            style='Custom.TLabelframe',
            padding=15
        )
        images_frame.pack(fill='both', expand=True, pady=10)

        # Container per le 4 facce
        faces_grid = tk.Frame(images_frame, bg="#2d2d44")
        faces_grid.pack(fill='both', expand=True)

        self.face_frames = {}
        self.face_labels = {}
        self.face_buttons = {}
        self.face_status = {}

        # Creazione dei 4 quadranti per le immagini
        for i, face_name in enumerate(self.face_images.keys(), 1):
            face_frame = tk.Frame(
                faces_grid, 
                bg="#3a3a5a", 
                relief='ridge', 
                bd=2,
                highlightbackground='#4ecdc4',
                highlightthickness=1
            )
            face_frame.grid(
                row=(i-1)//2, 
                column=(i-1)%2, 
                padx=10, 
                pady=10, 
                sticky='nsew'
            )
            face_frame.grid_propagate(False)
            face_frame.config(width=350, height=250)

            # Configurazione del grid per espandere uniformemente
            faces_grid.grid_rowconfigure((i-1)//2, weight=1)
            faces_grid.grid_columnconfigure((i-1)%2, weight=1)

            # Titolo faccia
            title_label = tk.Label(
                face_frame,
                text=f"🎪 {face_name}",
                font=("Segoe UI", 12, "bold"),
                bg="#3a3a5a",
                fg="white"
            )
            title_label.pack(pady=8)

            # Label per l'immagine
            image_label = tk.Label(
                face_frame,
                text="🖼️ Clicca per caricare\nimmagine",
                bg="#2d2d44",
                fg="#a5b3c1",
                width=30,
                height=8,
                relief='sunken',
                font=("Segoe UI", 9),
                cursor="hand2"
            )
            image_label.pack(padx=10, pady=5)
            image_label.bind("<Button-1>", lambda e, fn=face_name: self.load_image(fn))

            # Label stato
            status_label = tk.Label(
                face_frame,
                text="⏳ In attesa",
                font=("Segoe UI", 9),
                bg="#3a3a5a",
                fg="#a5b3c1"
            )
            status_label.pack(pady=3)

            # Bottone caricamento
            load_btn = ttk.Button(
                face_frame,
                text=f"📁 Carica {face_name}",
                style='Custom.TButton',
                command=lambda fn=face_name: self.load_image(fn)
            )
            load_btn.pack(pady=5)

            self.face_frames[face_name] = face_frame
            self.face_labels[face_name] = image_label
            self.face_buttons[face_name] = load_btn
            self.face_status[face_name] = status_label

        # Frame controlli
        controls_frame = ttk.LabelFrame(
            scrollable_frame, 
            text="🎮 CONTROLLI DI ANALISI", 
            style='Custom.TLabelframe',
            padding=15
        )
        controls_frame.pack(fill='x', pady=10)

        controls_inner = tk.Frame(controls_frame, bg="#2d2d44")
        controls_inner.pack(fill='x')

        self.predict_button = ttk.Button(
            controls_inner,
            text="🚀 AVVIA ANALISI IA",
            style='Green.TButton',
            command=self.start_prediction,
            state="disabled"
        )
        self.predict_button.pack(side='left', padx=5, pady=10)

        ttk.Button(
            controls_inner,
            text="🗑️ PULISCI TUTTO",
            style='Red.TButton',
            command=self.clear_all
        ).pack(side='left', padx=5, pady=10)

        # Progress bar moderna
        self.progress = ttk.Progressbar(
            controls_inner, 
            mode='indeterminate',
            style='Custom.Horizontal.TProgressbar'
        )
        style.configure('Custom.Horizontal.TProgressbar', 
                       background='#4ecdc4',
                       troughcolor='#2d2d44')
        self.progress.pack(side='right', fill='x', expand=True, padx=10, pady=10)

        # Frame risultati
        results_frame = ttk.LabelFrame(
            scrollable_frame, 
            text="📊 RISULTATI ANALISI IN TEMPO REALE", 
            style='Custom.TLabelframe',
            padding=15
        )
        results_frame.pack(fill='both', expand=True, pady=10)

        # Text widget con stile moderno
        self.results_text = tk.Text(
            results_frame,
            height=12,
            font=("Consolas", 10),
            wrap="word",
            bg="#1a1a2e",
            fg="#e2e2e2",
            insertbackground="#4ecdc4",
            selectbackground="#4ecdc4",
            padx=10,
            pady=10,
            relief='flat'
        )

        # Configurazione tag per colori
        self.results_text.tag_configure('SUCCESS', foreground='#6bcf7f')
        self.results_text.tag_configure('WARNING', foreground='#ffd93d')
        self.results_text.tag_configure('ERROR', foreground='#ff6b6b')
        self.results_text.tag_configure('INFO', foreground='#4ecdc4')
        self.results_text.tag_configure('HIGHLIGHT', foreground='#ff9ff3')

        scrollbar_text = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar_text.set)

        self.results_text.pack(side='left', fill='both', expand=True)
        scrollbar_text.pack(side='right', fill='y')

        # Footer
        footer_frame = tk.Frame(scrollable_frame, bg="#2d2d44", height=40)
        footer_frame.pack(fill='x', pady=10)
        footer_frame.pack_propagate(False)

        footer_label = tk.Label(
            footer_frame,
            text="🔬 BriCANet - Sistema avanzato di analisi corrosione | Versione 2.0",
            font=("Segoe UI", 9),
            fg="#a5b3c1",
            bg="#2d2d44"
        )
        footer_label.pack(anchor='center', pady=10)

        # Pack canvas e scrollbar
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Bind per lo scroll con mouse wheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def load_model(self):
        """Carica il modello BriCANet"""
        file_path = filedialog.askopenfilename(
            title="Seleziona il modello BriCANet (.h5)",
            filetypes=[("Modelli H5", "*.h5"), ("Tutti i file", "*.*")]
        )
        
        if file_path:
            try:
                self.model = load_model(file_path)
                self.model_status.config(
                    text=f"✅ Modello caricato: {os.path.basename(file_path)}", 
                    fg="#6bcf7f"
                )
                self.update_predict_button_state()
                self.log_result(f"✅ MODELLO IA CARICATO: {os.path.basename(file_path)}", 'SUCCESS')
            except Exception as e:
                messagebox.showerror("Errore Critico", f"Errore nel caricamento del modello:\n{str(e)}")
                self.log_result(f"❌ ERRORE CARICAMENTO MODELLO: {str(e)}", 'ERROR')

    def load_image(self, face_name):
        """Carica un'immagine per una faccia specifica"""
        file_path = filedialog.askopenfilename(
            title=f"Seleziona immagine per {face_name}",
            filetypes=[
                ("Immagini supportate", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("Tutti i file", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Carica e ridimensiona l'immagine
                pil_image = Image.open(file_path)
                pil_image.thumbnail((250, 180), Image.Resampling.LANCZOS)
                
                # Aggiungi bordi arrotondati
                mask = Image.new("L", pil_image.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.rounded_rectangle([(0, 0), pil_image.size], 15, fill=255)
                
                result = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
                result.paste(pil_image, (0, 0), mask=mask)
                photo = ImageTk.PhotoImage(result)
                
                # Aggiorna l'interfaccia
                self.face_labels[face_name].config(
                    image=photo, 
                    text="",
                    bg="#2d2d44"
                )
                self.face_labels[face_name].image = photo
                
                self.face_image_paths[face_name] = file_path
                self.face_images[face_name] = file_path
                self.face_status[face_name].config(
                    text="✅ Immagine caricata", 
                    fg="#6bcf7f"
                )
                
                self.update_predict_button_state()
                self.log_result(f"📁 IMMAGINE CARICATA: {face_name} - {os.path.basename(file_path)}", 'INFO')
                
            except Exception as e:
                messagebox.showerror("Errore", f"Errore nel caricamento dell'immagine:\n{str(e)}")
                self.log_result(f"❌ ERRORE IMMAGINE {face_name}: {str(e)}", 'ERROR')

    def update_predict_button_state(self):
        """Aggiorna lo stato del bottone di predizione"""
        if self.model and all(img is not None for img in self.face_images.values()):
            self.predict_button.config(state="normal")
            self.log_result("🎯 TUTTO PRONTO! Clicca 'AVVIA ANALISI IA' per iniziare", 'HIGHLIGHT')
        else:
            self.predict_button.config(state="disabled")

    def fuse_color_spaces(self, image):
        """Fusione avanzata degli spazi colore"""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        
        # Equalizzazione dell'istogramma per migliorare il contrasto
        hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
        
        # Fusione avanzata con pesi ottimali
        h, s, v = cv2.split(hsv_image)
        l, a, b = cv2.split(lab_image)
        
        # Fusione pesata per enfatizzare caratteristiche di corrosione
        fused_image = cv2.merge((
            cv2.addWeighted(h, 0.4, l, 0.6, 0),
            cv2.addWeighted(s, 0.5, a, 0.5, 0),
            cv2.addWeighted(v, 0.6, b, 0.4, 0)
        ))
        
        return fused_image

    def sliding_window(self, image, window_size=(224, 224), step_size=112):
        """Genera patch usando sliding window con overlap ottimale"""
        patches = []
        h, w, _ = image.shape
        
        for y in range(0, h - window_size[1] + 1, step_size):
            for x in range(0, w - window_size[0] + 1, step_size):
                patch = image[y:y + window_size[1], x:x + window_size[0]]
                patches.append(patch)
        
        return patches

    def preprocess_new_image(self, image_path):
        """Preprocessa una nuova immagine con tecniche avanzate"""
        try:
            # Carica e ridimensiona mantenendo aspect ratio
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Applica fusione spazi colore
            fused_image = self.fuse_color_spaces(image)
            
            # Genera patch
            patches = self.sliding_window(fused_image)
            
            # Normalizzazione avanzata
            processed_patches = []
            for patch in patches:
                # Normalizza tra 0 e 1
                normalized = patch / 255.0
                # Applica correzione gamma leggera
                normalized = np.power(normalized, 0.9)
                processed_patches.append(normalized)
            
            return np.array(processed_patches)
            
        except Exception as e:
            raise Exception(f"Errore preprocessing: {str(e)}")

    def predict_single_image(self, image_path, face_name):
        """Effettua la predizione su una singola immagine"""
        try:
            patches = self.preprocess_new_image(image_path)
            
            if len(patches) == 0:
                return {'face': face_name, 'error': 'Nessuna patch generata'}
            
            predictions = self.model.predict(patches, verbose=0)
            avg_prediction = np.mean(predictions, axis=0)
            
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
            return {'face': face_name, 'error': str(e)}

    def start_prediction(self):
        """Avvia la predizione in un thread separato"""
        if not all(img is not None for img in self.face_images.values()):
            messagebox.showwarning("Attenzione", "Carica tutte e 4 le immagini prima di procedere!")
            return
            
        threading.Thread(target=self.run_prediction, daemon=True).start()

    def run_prediction(self):
        """Esegue la predizione su tutte le immagini"""
        try:
            self.progress.start(20)
            self.predict_button.config(state="disabled")
            
            self.log_result("🚀 INIZIO ANALISI IA BRICANET...", 'HIGHLIGHT')
            self.log_result("=" * 70, 'INFO')
            
            results = []
            
            for face_name, image_path in self.face_image_paths.items():
                if image_path:
                    self.face_status[face_name].config(text="🔍 Analizzando...", fg="#ffd93d")
                    self.log_result(f"🔍 ANALISI {face_name} in corso...", 'INFO')
                    
                    result = self.predict_single_image(image_path, face_name)
                    results.append(result)
                    
                    if 'error' in result:
                        self.face_status[face_name].config(text="❌ Errore", fg="#ff6b6b")
                        self.log_result(f"❌ ERRORE {face_name}: {result['error']}", 'ERROR')
                    else:
                        color = self.class_colors[result['predicted_class']]
                        self.face_status[face_name].config(
                            text=f"✅ {result['predicted_class']} ({result['confidence']*100:.1f}%)",
                            fg=color
                        )
                        self.log_result(f"✅ {face_name} completata!", 'SUCCESS')
            
            self.display_detailed_results(results)
            
        except Exception as e:
            self.log_result(f"❌ ERRORE GENERALE: {str(e)}", 'ERROR')
            messagebox.showerror("Errore", f"Errore durante l'analisi:\n{str(e)}")
        
        finally:
            self.progress.stop()
            self.predict_button.config(state="normal")

    def display_detailed_results(self, results):
        """Visualizza i risultati dettagliati con formattazione avanzata"""
        self.log_result("\n📊 RISULTATI DETTAGLIATI ANALISI", 'HIGHLIGHT')
        self.log_result("=" * 70, 'INFO')
        
        summary = {'Alto': 0, 'Medio': 0, 'Basso': 0}
        confidence_sum = 0
        valid_results = 0
        
        for result in results:
            if 'error' not in result:
                face = result['face']
                pred_class = result['predicted_class']
                confidence = result['confidence']
                probabilities = result['probabilities']
                num_patches = result['num_patches']
                
                # Colore in base alla classe
                tag = 'ERROR' if pred_class == 'Alto' else 'WARNING' if pred_class == 'Medio' else 'SUCCESS'
                
                self.log_result(f"\n🎪 {face}:", 'INFO')
                self.log_result(f"   📈 Classe predetta: {pred_class}", tag)
                self.log_result(f"   🎯 Confidenza: {confidence:.4f} ({confidence*100:.2f}%)", tag)
                self.log_result(f"   🔬 Patch analizzate: {num_patches}", 'INFO')
                self.log_result(f"   📊 Distribuzione probabilità:", 'INFO')
                
                # Barre visuali per le probabilità
                for i, class_name in enumerate(self.class_names):
                    prob = probabilities[i]
                    bar_length = int(prob * 30)
                    bar = "█" * bar_length + " " * (30 - bar_length)
                    color_tag = 'ERROR' if class_name == 'Alto' else 'WARNING' if class_name == 'Medio' else 'SUCCESS'
                    self.log_result(f"     {class_name:6} {prob*100:6.2f}% [{bar}]", color_tag)
                
                summary[pred_class] += 1
                confidence_sum += confidence
                valid_results += 1
        
        # Statistiche finali
        if valid_results > 0:
            avg_confidence = confidence_sum / valid_results
            self.log_result("\n📈 STATISTICHE FINALI", 'HIGHLIGHT')
            self.log_result("=" * 50, 'INFO')
            
            for class_name in self.class_names:
                count = summary[class_name]
                percentage = (count / valid_results * 100) if valid_results > 0 else 0
                tag = 'ERROR' if class_name == 'Alto' else 'WARNING' if class_name == 'Medio' else 'SUCCESS'
                self.log_result(f"{class_name:6}: {count:2d}/{valid_results} ({percentage:5.1f}%)", tag)
            
            self.log_result(f"\n🎯 Confidenza media: {avg_confidence*100:.2f}%", 
                          'ERROR' if avg_confidence < 0.6 else 'WARNING' if avg_confidence < 0.8 else 'SUCCESS')
            
            # Classe predominante
            predominant = max(summary, key=summary.get)
            self.log_result(f"🏆 Classe predominante: {predominant}", 
                          'ERROR' if predominant == 'Alto' else 'WARNING' if predominant == 'Medio' else 'SUCCESS')
            
            # # Raccomandazione
            # self.log_result("\n💡 RACCOMANDAZIONE:", 'HIGHLIGHT')
            # if predominant == 'Alto':
            #     self.log_result("   ❗ INTERVENTO URGENTE richiesto! Alto rischio corrosione", 'ERROR')
            # elif predominant == 'Medio':
            #     self.log_result("   ⚠️  Monitoraggio intensivo raccomandato", 'WARNING')
            # else:
            #     self.log_result("   ✅ Situazione sotto controllo. Manutenzione ordinaria", 'SUCCESS')
        
        self.log_result("\n" + "=" * 70, 'INFO')
        self.log_result("✅ ANALISI COMPLETATA CON SUCCESSO!", 'SUCCESS')

    def log_result(self, message, tag='INFO'):
        """Aggiunge un messaggio ai risultati con formattazione colorata"""
        def update_text():
            self.results_text.insert(tk.END, message + "\n", tag)
            self.results_text.see(tk.END)
            self.root.update()
        
        self.root.after(0, update_text)

    def clear_all(self):
        """Pulisce tutti i dati e resetta l'interfaccia"""
        # Reset immagini
        for face_name in self.face_images.keys():
            self.face_images[face_name] = None
            self.face_image_paths[face_name] = None
            self.face_labels[face_name].config(
                image="",
                text="🖼️ Clicca per caricare\nimmagine",
                bg="#2d2d44",
                fg="#a5b3c1"
            )
            self.face_labels[face_name].image = None
            self.face_status[face_name].config(text="⏳ In attesa", fg="#a5b3c1")
        
        # Reset risultati
        self.results_text.delete(1.0, tk.END)
        
        # Reset stato
        self.update_predict_button_state()
        
        self.log_result("🧹 Sistema pulito e pronto per una nuova analisi...", 'INFO')

def main():
    # Aggiungi il percorso per PIL
    try:
        from PIL import Image, ImageTk, ImageDraw
    except ImportError:
        print("PIL non disponibile. Installare Pillow: pip install Pillow")
        return

    root = tk.Tk()
    app = CorrosionClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()