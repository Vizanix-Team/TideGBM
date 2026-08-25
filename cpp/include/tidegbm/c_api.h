#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void* TideHandle;

int tide_create(TideHandle* out);
void tide_free(TideHandle h);

/* cfg packed as key/value is done from Python; C API uses defaults + setters */
int tide_set_int(TideHandle h, const char* key, int value);
int tide_set_double(TideHandle h, const char* key, double value);

int tide_fit(TideHandle h, const double* X, const double* y, int n, int n_features,
             const long long* timestamps, const double* weight, const int* is_cat,
             int iid, int horizon);

int tide_predict(TideHandle h, const double* X, int n, int n_features, double* out);
int tide_predict_raw(TideHandle h, const double* X, int n, int n_features, double* out);
int tide_predict_binned(TideHandle h, const unsigned short* xb, int n, double* out);
int tide_save_json(TideHandle h, const char* path);

int tide_save(TideHandle h, const char* path);
int tide_load(TideHandle* out, const char* path);

const char* tide_last_error(void);

#ifdef __cplusplus
}
#endif
