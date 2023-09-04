import Head from 'next/head';
import { useState } from 'react';
import * as tf from '@tensorflow/tfjs';

export default function Home() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [result, setResult] = useState(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
    setResult(null);
  };
  function medianFilter(image, filterSize) {
    const width = image.shape[0];
    const height = image.shape[1];
    const channels = image.shape[2];
  
    const filteredData = new Uint8Array(width * height * channels);
  
    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height; y++) {
        for (let c = 0; c < channels; c++) {
          const neighbors = [];
          for (let i = -filterSize; i <= filterSize; i++) {
            for (let j = -filterSize; j <= filterSize; j++) {
              const nx = x + i;
              const ny = y + j;
  
              if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const neighborIndex = (ny * width + nx) * channels + c;
                neighbors.push(image.data[neighborIndex]);
              }
            }
          }
          neighbors.sort((a, b) => a - b);
          const medianIndex = Math.floor(neighbors.length / 2);
          const medianValue = neighbors[medianIndex];
  
          const pixelIndex = (y * width + x) * channels + c;
          filteredData[pixelIndex] = medianValue;
        }
      }
    }
  
    return tf.tensor3d(filteredData, [width, height, channels]);
  }  
  function equalizeHist(image) {
    const flat = image.dataSync();
    const histogram = new Array(256).fill(0);
    const equalizedData = tf.clone(image);
    for (let i = 0; i < flat.length; i++) {
      histogram[flat[i]]++;
    }
    const cdf = histogram.slice();
    for (let i = 1; i < cdf.length; i++) {
      cdf[i] += cdf[i - 1];
    }
    for (let i = 0; i < cdf.length; i++) {
      cdf[i] = (cdf[i] / cdf[cdf.length - 1]) * 255;
    }
    const equalized = equalizedData.dataSync();
    for (let i = 0; i < equalized.length; i++) {
      equalized[i] = cdf[equalized[i]];
    }
  
    return tf.tensor(equalized, image.shape);
  }
  

  const preprocessImage = (imageElement, targetWidth, targetHeight) => {
    let imageTensor = tf.browser.fromPixels(imageElement);
    const inputAspectRatio = imageElement.width / imageElement.height;
    const targetAspectRatio = 1;
    let cropWidth, cropHeight, offsetX, offsetY;
    if (inputAspectRatio > targetAspectRatio) {
      cropWidth = imageElement.height * targetAspectRatio;
      cropHeight = imageElement.height;
      offsetX = (imageElement.width - cropWidth) / 2;
      offsetY = 0;
    } else {
      cropWidth = imageElement.width;
      cropHeight = imageElement.width / targetAspectRatio;
      offsetX = 0;
      offsetY = (imageElement.height - cropHeight) / 2;
    }
    imageTensor = imageTensor.expandDims(0);
    const boxes = [[offsetY / imageElement.height, offsetX / imageElement.width, (offsetY + cropHeight) / imageElement.height, (offsetX + cropWidth) / imageElement.width]];
    const boxIndices = [0];
    const cropSize = [targetHeight, targetWidth];
    const croppedImage = tf.image.cropAndResize(imageTensor, boxes, boxIndices, cropSize);
    const rank3CroppedImage = croppedImage.squeeze([0]);
    const smoothedImage = medianFilter(rank3CroppedImage, 3);
    const equalizedImage = equalizeHist(smoothedImage);
    const preprocessedImage = equalizedImage.toFloat().div(255).expandDims(0);
    return preprocessedImage;
  };  
  

  const handleSubmit = async() => {
    const model = await tf.loadLayersModel('/ts-js/model.json');

    if (selectedImage) {
      const imageElement = document.getElementById('uploaded-image');
      if (imageElement) {
        const targetWidth = 224;
        const targetHeight = 224;
        const preprocessedImage = preprocessImage(imageElement, targetWidth, targetHeight);

        const prediction = model.predict(preprocessedImage);
        const isPositive = prediction.dataSync[0] > prediction.dataSync[1];
        setResult(isPositive ? 'DR' : 'NO_DR');

        preprocessedImage.dispose();
      } else {
        console.error("Image element with ID 'uploaded-image' not found.");
      }
    }
  };

  const isSubmitDisabled = selectedImage === null;

  return (
    <>
      <Head>
        <title>DR Detection</title>
        <meta name="description" content="Diabetic Retinopathy Detection" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main style={styles.container}>
        <div style={styles.card}>
          <h1 style={styles.title}>DR Detection</h1>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            style={styles.fileInput}
          />
          {selectedImage && (
            <div style={styles.imageContainer}>
              <img
                id='uploaded-image'
                src={selectedImage}
                alt="Uploaded Image"
                style={styles.uploadedImage}
              />
              <button onClick={handleRemoveImage} style={styles.removeButton}>
                Remove
              </button>
            </div>
          )}
          <div style={styles.buttonContainer}>
            <button
              onClick={handleSubmit}
              style={isSubmitDisabled ? styles.submitButtonDisabled : styles.submitButton}
              disabled={isSubmitDisabled}
            >
              Submit
            </button>
          </div>
          {result !== null && (
            <div style={styles.resultContainer}>
              <h2>Result:</h2>
              <p style={result === 'DR' ? styles.resultPositive : styles.resultNegative}>
                {result === 'DR' ? 'Retinopathy Positive' : 'Retinopathy Negative'}
              </p>
            </div>
          )}
        </div>
      </main>
    </>
  );
}

const styles = {
  container: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: '100vh',
    background: '#f5f5f5',
  },
  card: {
    background: '#fff',
    borderRadius: '12px',
    padding: '24px',
    boxShadow: '0px 8px 16px rgba(0, 0, 0, 0.2)',
    width: '300px',
    textAlign: 'center',
  },
  title: {
    fontSize: '28px',
    marginBottom: '20px',
    color: '#333',
  },
  fileInput: {
    marginBottom: '20px',
    padding: '8px',
    borderRadius: '6px',
    border: '1px solid #ddd',
    width: '100%',
  },
  imageContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  uploadedImage: {
    maxWidth: '100%',
    maxHeight: '200px',
    borderRadius: '8px',
    boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.2)',
  },
  removeButton: {
    marginTop: '10px',
    padding: '8px 16px',
    borderRadius: '6px',
    background: '#ff6347',
    color: '#fff',
    border: 'none',
    cursor: 'pointer',
  },
  buttonContainer: {
    marginTop: '20px',
  },
  submitButton: {
    padding: '12px 24px',
    borderRadius: '6px',
    background: '#007bff',
    color: '#fff',
    border: 'none',
    cursor: 'pointer',
  },
  submitButtonDisabled: {
    padding: '12px 24px',
    borderRadius: '6px',
    background: '#ccc',
    color: '#666',
    border: 'none',
    cursor: 'not-allowed',
  },
    resultContainer: {
    marginTop: '20px',
    padding: '10px',
    border: '1px solid #ccc',
    borderRadius: '6px',
    textAlign: 'center',
  },
  resultPositive: {
    color: 'red',
    textShadow: '0 0 5px rgba(255, 0, 0, 0.5)',
  },
  resultNegative: {
    color: 'green',
    textShadow: '0 0 5px rgba(0, 128, 0, 0.5)',
  },
};