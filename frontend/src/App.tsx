import "./App.css";
import { useState } from "react";

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<{
    species: string;
    probability: number;
  } | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setPrediction(data);
  };

  return (
    <>
      <head>
        <title>Nate's Pet Classifier</title>
      </head>
      <div className="container">
        <h1 style={{ fontSize: "100px" }}>Nate's Pet Classifier</h1>
        <input type="file" onChange={handleFileChange} />
        {preview && <img src={preview} alt="preview" width={200} />}
        <button onClick={handleUpload}>Predict</button>

        {prediction && (
          <div>
            <h1>Species: {prediction.species}</h1>
            <h1>Probability: {prediction.probability}%</h1>
          </div>
        )}
      </div>
    </>
  );
}

export default App;
