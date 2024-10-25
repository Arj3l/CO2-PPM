from preprocessing import load_data, clean_data, preprocess_data
from visualization import plot_co2_levels, plot_co2_by_year
from model import split_data, train_models, evaluate_all_models

def main():
    # Load and clean data
    df = load_data()
    df = clean_data(df)

    # Visualize data
    plot_co2_levels(df)
    plot_co2_by_year(df)

    # Preprocess data
    X, y = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate models
    evaluate_all_models(models, X_test, y_test)

if __name__ == "__main__":
    main()