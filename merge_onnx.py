import onnx

# Load model from public/model/
model = onnx.load('public/model/canvasmath.onnx')

# Save it back as a single standalone file with external data disabled
onnx.save(model, 'public/model/canvasmath.onnx', save_as_external_data=False)
print("Model successfully merged into a single file!")