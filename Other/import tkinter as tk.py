import tkinter as tk
from tkinter import ttk

class CircleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cercle Ajustable")
        
        # Canvas pour dessiner le cercle
        self.canvas = tk.Canvas(root, width=400, height=400, bg='white')
        self.canvas.pack(pady=10)
        
        # Molette pour ajuster le rayon
        self.radius_scale = ttk.Scale(
            root,
            from_=10,
            to=150,
            orient='horizontal',
            length=300,
            command=self.update_circle
        )
        self.radius_scale.set(50)  # Rayon initial
        self.radius_scale.pack(pady=10)
        
        # Dessiner le cercle initial
        self.circle = None
        self.draw_circle(50)
        
    def draw_circle(self, radius):
        # Effacer l'ancien cercle
        if self.circle:
            self.canvas.delete(self.circle)
        
        # Calculer les coordonnées du cercle
        center_x = 200
        center_y = 200
        x1 = center_x - radius
        y1 = center_y - radius
        x2 = center_x + radius
        y2 = center_y + radius
        
        # Dessiner le nouveau cercle
        self.circle = self.canvas.create_oval(x1, y1, x2, y2, fill='yellow')
        
    def update_circle(self, value):
        # Mettre à jour le cercle quand la molette est déplacée
        radius = float(value)
        self.draw_circle(radius)

# Créer et lancer l'application
root = tk.Tk()
app = CircleApp(root)
root.mainloop()
