from vgg_anomaly import AnomalyPredictor

if __name__ == "__main__":
    anomaly = AnomalyPredictor()
    anomaly.init_data()
    anomaly.create_model()
    anomaly.train_model()
    anomaly.set_threshold()
    # anomaly.show_plot()
