from vgg_anomaly import AnomalyPredictor

if __name__ == "__main__":
    anomaly = AnomalyPredictor()
    y_test,y_pred=anomaly.anomaly_function()
    print("Test: ",y_test)
    print("Pred: ",y_pred)
    # anomaly.create_model()
    # anomaly.load_weights()
    # anomaly.get_threshold()
    # for img in anomaly.test_image_list:
    #     print(
    #         "Is %s an anomaly: " % img,
    #         anomaly.IsImageHasAnomaly(
    #             os.path.join(anomaly.test_data_dir, "cogs", img), img))
