import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.applications.densenet import preprocess_input as densenet_preprocess
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from keras.utils import img_to_array

DenseNet_model = tf.keras.models.load_model('chest_CT_SCAN-DenseNet201.keras', compile=False)
ResNet_model = tf.keras.models.load_model('chest_CT_SCAN-ResNet50.keras', compile=False)
class_dict = {0: "Adenocarcinoma", 1: "Large Cell Carcinoma", 2: "Normal", 3: "Squamous Cell Carcinoma"}


def preprocess_image(image_path, target_size=(460, 460)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    arr = img_to_array(img)
    arr = arr.reshape((1,) + arr.shape)
    return arr


def ensemble_predict(image_path):
    img_arr = preprocess_image(image_path)

    img_resnet = resnet_preprocess(img_arr.copy())
    img_densenet = densenet_preprocess(img_arr.copy())

    pred_resnet = ResNet_model.predict(img_resnet)[0]
    pred_densenet = DenseNet_model.predict(img_densenet)[0]

    combined = [
                   (prob, "ResNet50", i) for i, prob in enumerate(pred_resnet)
               ] + [
                   (prob, "DenseNet201", i) for i, prob in enumerate(pred_densenet)
               ]

    best_prob, model_used, pred_class = max(combined, key=lambda x: x[0])

    return pred_resnet, pred_densenet, class_dict[pred_class], model_used

root = tk.Tk()
root.title("Lung Cancer Image Classifier")
root.geometry("1000x800")

header_frame = tk.Frame(root)
header_frame.pack(fill="x", pady=(10, 0))

tk.Label(
    header_frame,
    text="Lung Cancer Image Classifier",
    font=("Helvetica", 16, "bold")
).pack(pady=(0, 5), side="top")

tk.Button(
    header_frame,
    text="Open Image",
    command=lambda: open_image()
).pack(pady=(0, 10), side="top")

placeholder_frame = tk.Frame(root)
placeholder_frame.pack(fill="both", expand=True)

placeholder_text_container = tk.Frame(placeholder_frame)
placeholder_text_container.pack(fill="both", expand=True)

placeholder_label = tk.Label(
    placeholder_text_container,
    text="Interpreting radiographic patterns to distinguish adenocarcinoma, large cell, and squamous cell carcinoma by "
         "leveraging deep learning to support evidence-based diagnostics.",
    font=("Arial", 14),
    wraplength=500,
    justify="center"
)
placeholder_label.place(relx=0.5, rely=0.5, anchor="center")
content_frame = tk.Frame(root)

top_frame = tk.Frame(content_frame)
top_frame.pack(pady=10)

image_frame = tk.Frame(top_frame, width=400, height=400, bd=2, relief="sunken")
image_frame.pack(side=tk.LEFT, padx=20, pady=10)
image_frame.pack_propagate(False)
image_label = tk.Label(image_frame)
image_label.pack(fill='both', expand=True)

right_frame = tk.Frame(top_frame)
right_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=True)

image_info_label = tk.Label(content_frame, text="", font=("Arial", 10, "italic"))
image_info_label.pack()
status_label = tk.Label(content_frame, text="", font=("Arial", 12, "bold"))
status_label.pack(pady=5)

def create_model_frame(model_name):
    frame = tk.LabelFrame(right_frame, text=model_name, padx=10, pady=10)
    bars = []
    labels = []
    for i, cls in class_dict.items():
        label = tk.Label(frame, text=f"{cls}: 0.00%")
        label.pack(anchor='w', pady=(5, 0))
        bar = ttk.Progressbar(
            frame,
            orient='horizontal',
            length=200,
            mode='determinate',
            maximum=100
        )
        bar.pack(fill='x', pady=(0, 5))
        bars.append(bar)
        labels.append(label)
    return frame, bars, labels

resnet_frame, resnet_bars, resnet_labels = create_model_frame("ResNet50")
densenet_frame, densenet_bars, densenet_labels = create_model_frame("DenseNet201")
resnet_frame.pack(pady=5)
densenet_frame.pack(pady=5)

def open_image():
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )
    if file_path:
        try:
            placeholder_frame.pack_forget()
            content_frame.pack()

            loading_label = tk.Label(content_frame, text="Loading, please wait...", font=("Arial", 12, "italic"),
                                     fg="blue")
            loading_label.pack(pady=10)
            root.update()

            img = Image.open(file_path)
            img.thumbnail((350, 350))
            photo = ImageTk.PhotoImage(img)
            image_label.config(image=photo)
            image_label.image = photo
            image_info_label.config(
                text=f"File: {os.path.basename(file_path)} | {img.size[0]}x{img.size[1]} | {os.path.getsize(file_path) // 1024} KB"
            )

            pred_resnet, pred_densenet, final_class, model_used = ensemble_predict(file_path)
            status_label.config(text=f"Final Prediction: {final_class}")

            for i, cls in class_dict.items():
                resnet_bars[i]['value'] = pred_resnet[i] * 100
                densenet_bars[i]['value'] = pred_densenet[i] * 100
                resnet_labels[i].config(text=f"{cls}: {pred_resnet[i] * 100:.2f}%")
                densenet_labels[i].config(text=f"{cls}: {pred_densenet[i] * 100:.2f}%")

            loading_label.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict: {e}")
disclaimer_frame = tk.Frame(root)
disclaimer_frame.pack(side="bottom", fill="x")

disclaimer_label = tk.Label(
    disclaimer_frame,
    text="Disclaimer: This AI model may make mistakes. Always consult a qualified medical professional for diagnosis an"
         "d treatment.",
    font=("Arial", 10, "italic"),
    fg="red",
    wraplength=1000,
    justify="center"
)
disclaimer_label.pack(pady=5)

root.mainloop()
