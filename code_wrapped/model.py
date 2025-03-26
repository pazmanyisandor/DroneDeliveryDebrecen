import tkinter as tk
from tkinter import messagebox
import tkintermapview
import pandas as pd
import numpy as np
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Data and Train the Model
df = pd.read_csv("output.csv", sep="\t")
df['Coordinates'] = df['Coordinates'].apply(ast.literal_eval)
df['lat'] = df['Coordinates'].apply(lambda coord: coord[0])
df['lon'] = df['Coordinates'].apply(lambda coord: coord[1])
X = df[['lat', 'lon']]
y = df['fastest_mode']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.2f}")

def predict_transport_mode(coordinate):
    """
    Predict the best transportation mode for a given coordinate.
    
    Parameters:
      coordinate (tuple): A tuple (lat, lon) of the destination.
    
    Returns:
      mode (str): Predicted best transportation mode.
      confidence (float): Confidence of the prediction.
    """
    X_new = pd.DataFrame([coordinate], columns=['lat', 'lon'])
    pred_mode = clf.predict(X_new)[0]
    pred_proba = clf.predict_proba(X_new)[0]
    confidence = np.max(pred_proba)
    return pred_mode, confidence

# Build the Interactive Map Window
root = tk.Tk()
root.title("Select Destination on Map")
root.geometry("800x600")

selection_enabled = False

def enable_selection():
    global selection_enabled
    selection_enabled = True
    select_button.config(text="Click on Map...")

select_button = tk.Button(root, text="Select Location", command=enable_selection)
select_button.pack(pady=5)

map_widget = tkintermapview.TkinterMapView(root, width=800, height=550, corner_radius=0)
map_widget.set_position(47.5316, 21.6273)  # Coordinates for Debrecen
map_widget.set_zoom(12)
map_widget.pack(fill="both", expand=True)

def on_map_click(event):
    global selection_enabled
    if not selection_enabled:
        return

    map_widget.delete_all_marker()

    lat, lon = map_widget.convert_canvas_coords_to_decimal_coords(event.x, event.y)
    mode, conf = predict_transport_mode((lat, lon))
    
    marker_text = f"{mode} ({conf:.2f})"
    map_widget.set_marker(lat, lon, text=marker_text)
    
    selection_enabled = False
    select_button.config(text="Select Location")

map_widget.canvas.bind("<Button-1>", on_map_click)

root.mainloop()