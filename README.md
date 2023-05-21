# Disease Prediction API

This is an API for disease prediction based on symptoms. It uses a machine learning model trained on symptom data to predict the most likely disease given a set of symptoms. The API is built using Flask and relies on a trained model stored in a pickle file.

## Installation

1. Clone the repository: `git clone <repository-url>`
2. Navigate to the project directory: `cd <project-directory>`
3. Install the dependencies: `pip install -r requirements.txt`

## Usage

To start the server, run the following command:

```
python app.py
```

The server will start running at `http://localhost:5000`.

## API Endpoints

The following are the available API endpoints:

- `GET /` - Retrieves a welcome message.
- `POST /search` - Predicts the disease based on provided symptoms.

Please refer to the API documentation for detailed information on each endpoint.

## Dependencies

The following are the main dependencies used in this project:

- `Flask` - A lightweight web framework for Python.
- `pandas` - A data manipulation and analysis library.
- `numpy` - A numerical computing library.
- `scikit-learn` - A machine learning library.
- `flask_cors` - A Flask extension for handling Cross-Origin Resource Sharing (CORS) requests.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

---
