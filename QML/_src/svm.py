from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

data_url = "/home/fcsc/ccalvo/ML_meteo/utils/SCRIPTS/test/enero/buenos/semana.csv"
df = pd.read_csv(data_url)

#X = df.drop(['DATE','TIMESTAMP','RAINC','RAINNC','RAIN_CHE' ,'RAIN_WRF','RAINING_CHE','RAINING_WRF','RANGE_CHE','RANGE_WRF','RAINING_ERROR','RANGE_ERROR','YEAR','MONTH'], axis=1)
X = df.drop(["DATE","TIMESTAMP","XLAT","XLONG","HGT","RAINC","RAINNC","QVAPOR_500_0h","QVAPOR_500_3h","QVAPOR_500_6h","QVAPOR_500_9h","QVAPOR_500_12h","QVAPOR_500_15h","QVAPOR_500_18h","QVAPOR_500_21h","QVAPOR_700_0h","QVAPOR_700_3h","QVAPOR_700_6h","QVAPOR_700_9h","QVAPOR_700_12h","QVAPOR_700_15h","QVAPOR_700_18h","QVAPOR_700_21h","QVAPOR_850_0h","QVAPOR_850_3h","QVAPOR_850_6h","QVAPOR_850_9h","QVAPOR_850_12h","QVAPOR_850_15h","QVAPOR_850_18h","QVAPOR_850_21h","QCLOUD_500_0h","QCLOUD_500_3h","QCLOUD_500_6h","QCLOUD_500_9h","QCLOUD_500_12h","QCLOUD_500_15h","QCLOUD_500_18h","QCLOUD_500_21h","QCLOUD_700_0h","QCLOUD_700_3h","QCLOUD_700_6h","QCLOUD_700_9h","QCLOUD_700_12h","QCLOUD_700_15h","QCLOUD_700_18h","QCLOUD_700_21h","QCLOUD_850_0h","QCLOUD_850_3h","QCLOUD_850_6h","QCLOUD_850_9h","QCLOUD_850_12h","QCLOUD_850_15h","QCLOUD_850_18h","QCLOUD_850_21h","QRAIN_500_0h","QRAIN_500_3h","QRAIN_500_6h","QRAIN_500_9h","QRAIN_500_12h","QRAIN_500_15h","QRAIN_500_18h","QRAIN_500_21h","QRAIN_700_0h","QRAIN_700_3h","QRAIN_700_6h","QRAIN_700_9h","QRAIN_700_12h","QRAIN_700_15h","QRAIN_700_18h","QRAIN_700_21h","QRAIN_850_0h","QRAIN_850_3h","QRAIN_850_6h","QRAIN_850_9h","QRAIN_850_12h","QRAIN_850_15h","QRAIN_850_18h","QRAIN_850_21h","QICE_500_0h","QICE_500_3h","QICE_500_6h","QICE_500_9h","QICE_500_12h","QICE_500_15h","QICE_500_18h","QICE_500_21h","QICE_700_0h","QICE_700_3h","QICE_700_6h","QICE_700_9h","QICE_700_12h","QICE_700_15h","QICE_700_18h","QICE_700_21h","QICE_850_0h","QICE_850_3h","QICE_850_6h","QICE_850_9h","QICE_850_12h","QICE_850_15h","QICE_850_18h","QICE_850_21h","QSNOW_500_0h","QSNOW_500_3h","QSNOW_500_6h","QSNOW_500_9h","QSNOW_500_12h","QSNOW_500_15h","QSNOW_500_18h","QSNOW_500_21h","QSNOW_700_0h","QSNOW_700_3h","QSNOW_700_6h","QSNOW_700_9h","QSNOW_700_12h","QSNOW_700_15h","QSNOW_700_18h","QSNOW_700_21h","QSNOW_850_0h","QSNOW_850_3h","QSNOW_850_6h","QSNOW_850_9h","QSNOW_850_12h","QSNOW_850_15h","QSNOW_850_18h","QSNOW_850_21h","QGRAUP_500_0h","QGRAUP_500_3h","QGRAUP_500_6h","QGRAUP_500_9h","QGRAUP_500_12h","QGRAUP_500_15h","QGRAUP_500_18h","QGRAUP_500_21h","QGRAUP_700_0h","QGRAUP_700_3h","QGRAUP_700_6h","QGRAUP_700_9h","QGRAUP_700_12h","QGRAUP_700_15h","QGRAUP_700_18h","QGRAUP_700_21h","QGRAUP_850_0h","QGRAUP_850_3h","QGRAUP_850_6h","QGRAUP_850_9h","QGRAUP_850_12h","QGRAUP_850_15h","QGRAUP_850_18h","QGRAUP_850_21h","YEAR","MONTH","RAIN_WRF","RAIN_CHE","RAINING_WRF","RAINING_CHE","RANGE_WRF","RANGE_CHE","RAINING_ERROR","RANGE_ERROR"], axis=1)

y = df.RAINING_CHE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

svc = SVC()
svc.fit(X_train, y_train)
svc_score = svc.score(X_test, y_test)

print(svc_score)

classical_preds = svc.predict(X_test)
print(classification_report(y_test,classical_preds))

import joblib
# save the model to disk
print("Guardando modelo...")
filename = 'modelo_svm.joblib'
joblib.dump(svc, open(filename, 'wb'))
