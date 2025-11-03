#include <Arduino.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Adafruit_MLX90614.h>
#include "MAX30105.h"
#include "heartRate.h"
#include "spo2_algorithm.h"

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <WiFiManager.h>

#include <math.h>
#include <time.h>

// ========= FIREBASE (akunmu) =========
const String FIREBASE_HOST        = "heart-monitoring-20872-default-rtdb.asia-southeast1.firebasedatabase.app"; // TANPA https://
const String FIREBASE_WEB_API_KEY = "AIzaSyB56eLVWtBYt6EN3-grFMvoW3mY2zV6Q-I"; // dari config web-mu
const String DEVICE_ID            = "ESP32_Monitor_Jantung";

// ========= TOKEN AUTH (didapat via Anonymous Sign-In) =========
String g_idToken       = "";
String g_refreshToken  = "";
unsigned long g_tokenExpiryMs = 0; // millis() kapan token habis

// ========= NTP TIME =========
const char* ntpServer         = "pool.ntp.org";
const long  gmtOffset_sec     = 25200; // UTC+7
const int   daylightOffset_sec = 0;

// ========= KALIBRASI TENSI =========
const int KALIBRASI_SISTOLIK  = -30;
const int KALIBRASI_DIASTOLIK = -20;

// ========= PIN / SERIAL =========
#define TOMBOL_PIN 4
#define RXD2 16
#define TXD2 17

// ========= LCD & SENSOR =========
LiquidCrystal_I2C lcd(0x27, 20, 4);
Adafruit_MLX90614 mlx = Adafruit_MLX90614();
MAX30105 sensorDetak;

// ========= STATUS =========
bool wifiTerhubung = false;
bool firebaseSiap  = false;
bool ntpSinkron    = false;

// ========= BUFFER TENSI =========
char  buff[64];
bool  b_read, b_discard;
char  discard;
int   i, j = 0;
char  final_buff[64];
int   nilaiSistolik = 0, nilaiDiastolik = 0, nilaiDetakDarah = 0;
float suhuTubuh = 0;

// ========= DETAK & SpO2 =========
#define UKURAN_BUFFER 100
uint32_t bufferIR[UKURAN_BUFFER];
uint32_t bufferMerah[UKURAN_BUFFER];

const int durasi_ms = 30000; // 30 detik untuk BPM & SpO2
long  lastBeat = 0;
float beatsPerMinute = 0;
int   bpm_final = 0;

int32_t spo2, heartRate;
int8_t  spo2Valid, heartRateValid;
int     spo2Final = 0;
int     detakJantungFinal = 0;

// ========= FALLBACK DATA =========
float suhuFallback  = 36.5;
int   detakFallback = 75;
int   spo2Fallback  = 98;

// ========= WiFiManager =========
#define CONFIG_AP_SSID "ESP32-Monitor"
#define CONFIG_AP_PASS "12345678"
#define CONFIG_PORTAL_TIMEOUT_SEC 180

// ========= NAIVE BAYES (Gaussian) =========
// Kelas: 0=Normal, 1=Kurang Normal, 2=Berbahaya
enum Feat { F_TEMP = 0, F_HR = 1, F_SYS = 2, F_DIA = 3, F_SPO2 = 4, F_COUNT = 5 };
#define NB_NUM_CLASSES 3

const char* NB_LABELS[NB_NUM_CLASSES] = {
  "Normal", "Kurang Normal", "Berbahaya"
};

// ====== MODEL BARU (DARI KAMU) ======
// prior
float nbPrior[NB_NUM_CLASSES] = {
  0.6700088006034699f, // class 0 (Normal)
  0.2297428966557707f, // class 1 (Kurang Normal)
  0.10024830274075937f // class 2 (Berbahaya)
};

// means[class][feature]  ->  { F_TEMP, F_HR, F_SYS, F_DIA, F_SPO2 }
float nbMu[NB_NUM_CLASSES][F_COUNT] = {
  { -0.07768017968697753f, -0.16852068120982594f, -0.29651842332941514f, -0.24343302131314512f,  0.13412888197439346f }, // 0
  {  0.07750665724656132f,  0.15026563739746684f,  0.5226181358251887f,   0.3976752076789426f,  -0.09526679965422409f }, // 1
  {  0.34154992294821107f,  0.7819371954203101f,   0.7840746092801186f,   0.715615232733371f,   -0.6781228107777277f }  // 2
};

// std[class][feature]  ->  { F_TEMP, F_HR, F_SYS, F_DIA, F_SPO2 }
float nbSigma[NB_NUM_CLASSES][F_COUNT] = {
  { 0.7074134976187391f,  0.814113300118089f,   0.7458991908828791f,  0.689969703524953f,  0.7583105240809166f }, // 0
  { 1.2495883905626117f,  1.1375924641562356f,  1.1285177501658672f,  1.24557055454495f,   0.9826186958566245f }, // 1
  { 1.6976356544865374f,  1.314232868280896f,   1.2288754778958344f,  1.4028064616779172f, 1.8217961970591066f }  // 2
};

bool   nbLoaded = false;
String nbLabel  = "Normal"; // hasil label

// ====== STANDARDIZER (z-score) ======
float stdz_mean[F_COUNT] = { 36.8f, 80.0f, 120.0f, 80.0f, 98.0f };
float stdz_std [F_COUNT] = {  0.7f, 15.0f,  12.0f, 10.0f,  1.5f };

// ========= PROTOTIPE =========
void connectWiFiViaManager();
void setupNTPTime();
void setupFirebase();
void setupSensorSempurna();
void tampilkanStatus();
void ukurSuhuTubuh();
void ukurDetakJantungFixed();
void ukurKadarOksigen();
void ukurTekananDarahFixed();
void tampilkanHasil();
String getStatusKesehatan();
void kirimDataKeFirebaseFixed();
String getNTPTimestamp();
unsigned long getNTPTime();
int  hexToDec(char high, char low);
int  hexDigit(char c);
String withAuth(const String& path);

// NB helpers
void   standardizeFeatures(float z[F_COUNT], float rawTemp, int rawHR, int rawSYS, int rawDIA, int rawSpO2);
int    argmax3(float a, float b, float c);
String predictNB_fromZ(const float z[F_COUNT]);
String predictNB(float rawTemp, int rawHR, int rawSYS, int rawDIA, int rawSpO2);
bool   loadNBModelFromFirebase();
bool   loadStandardizerFromFirebase();

// Firebase Auth helpers
bool firebaseSignInAnon();
bool firebaseRefreshTokenIfNeeded();

// =========================================================
void setup() {
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, RXD2, TXD2);

  Wire.begin(21, 22);          // SDA=21, SCL=22 (ubah sesuai wiring kamu)
  Wire.setClock(100000);       // clock 100kHz
  Wire.setTimeOut(1000);       // timeout 1 detik

  pinMode(TOMBOL_PIN, INPUT_PULLUP);
  randomSeed(esp_random());

  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Monitor Kesehatan");
  lcd.setCursor(0, 1); lcd.print("Init...");

  Serial.println("\n=== MONITOR + Gaussian Naive Bayes ===");
  Serial.println("Kalibrasi tensi: SYS-15, DIA-10");

  connectWiFiViaManager();

  if (wifiTerhubung) {
    setupNTPTime();
    setupFirebase();

    if (firebaseSiap) {
      bool okModel = loadNBModelFromFirebase();   // akan override kalau file ada
      bool okStd   = loadStandardizerFromFirebase(); // opsional
      nbLoaded     = okModel;

      Serial.println(ntpSinkron ? "NTP: OK" : "NTP: Gagal");
      if (!okModel) Serial.println("Model NB: pakai default firmware");
      else          Serial.println("✅ Model NB termuat dari Firebase");

      if (!okStd)   Serial.println("Standardizer: pakai default firmware");
      else          Serial.println("✅ Standardizer termuat dari Firebase");
    }
  }

  setupSensorSempurna();
  tampilkanStatus();
}

void loop() {
  if (digitalRead(TOMBOL_PIN) == LOW) {
    delay(300);
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("Mulai Pengukuran");
    lcd.setCursor(0, 1); lcd.print("Mohon tunggu...");
    delay(1200);

    ukurSuhuTubuh();          delay(500);
    ukurDetakJantungFixed();  delay(500);
    ukurKadarOksigen();       delay(500);
    ukurTekananDarahFixed();  delay(500);

    // === PREDIKSI NB ===
    nbLabel = predictNB(
      suhuTubuh,
      detakJantungFinal,
      nilaiSistolik,
      nilaiDiastolik,
      spo2Final
    );

    tampilkanHasil();

    if (wifiTerhubung && firebaseSiap) {
      kirimDataKeFirebaseFixed();
    } else {
      Serial.println("Mode offline - data tidak dikirim");
      lcd.setCursor(0, 3); lcd.print("Mode: Offline ");
      delay(800);
    }

    tampilkanStatus();
    delay(600);
  }

  delay(40);
}

// ================= WIFI MANAGER (portal HP) =================
void connectWiFiViaManager() {
  unsigned long t0 = millis();
  bool mauReset = false;

  if (digitalRead(TOMBOL_PIN) == LOW) {
    while (millis() - t0 < 5000) {
      if (digitalRead(TOMBOL_PIN) == HIGH) break;
      delay(10);
    }
    if (millis() - t0 >= 5000) mauReset = true;
  }

  WiFi.mode(WIFI_STA);
  WiFiManager wm;

  if (mauReset) {
    wm.resetSettings();
    Serial.println("⚠ Reset WiFi creds. Buka portal config...");
  }

  wm.setConfigPortalTimeout(CONFIG_PORTAL_TIMEOUT_SEC);

  lcd.setCursor(0, 1); lcd.print("WiFi cfg via HP ");
  lcd.setCursor(0, 2); lcd.print("AP: ESP32-Monitor");

  bool res = wm.autoConnect(CONFIG_AP_SSID, CONFIG_AP_PASS);

  if (res) {
    wifiTerhubung = true;
    Serial.println("✅ WiFi terhubung!");
    Serial.print("IP: "); Serial.println(WiFi.localIP());

    lcd.setCursor(0, 1); lcd.print("WiFi: Terhubung ");
    lcd.setCursor(0, 2); lcd.print(WiFi.localIP().toString() + " ");
  } else {
    wifiTerhubung = false;
    Serial.println("❌ Gagal WiFi");
    lcd.setCursor(0, 1); lcd.print("WiFi: Gagal ");
    lcd.setCursor(0, 2); lcd.print("Buka portal lg ");
  }

  delay(600);
}

// ================= NTP =================
void setupNTPTime() {
  lcd.setCursor(0, 3); lcd.print("Sync NTP time... ");
  Serial.println("Sync NTP...");

  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  struct tm timeinfo;
  int attempts = 0;

  while (!getLocalTime(&timeinfo) && attempts < 10) {
    Serial.println("Menunggu NTP...");
    delay(800);
    attempts++;
  }

  if (getLocalTime(&timeinfo)) {
    ntpSinkron = true;
    Serial.println("✅ NTP sinkron");
    lcd.setCursor(0, 3); lcd.print("NTP: OK ");
  } else {
    ntpSinkron = false;
    Serial.println("⚠ NTP gagal, pakai millis()");
    lcd.setCursor(0, 3); lcd.print("NTP: Gagal ");
  }

  delay(500);
}

// ================= FIREBASE AUTH (Anonymous) =================
bool firebaseSignInAnon() {
  WiFiClientSecure client; client.setInsecure();
  HTTPClient http;

  String url = "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=" + FIREBASE_WEB_API_KEY;
  if (!http.begin(client, url)) {
    Serial.println("Auth: http.begin gagal");
    return false;
  }

  http.addHeader("Content-Type", "application/json");
  int code    = http.POST("{\"returnSecureToken\":true}");
  String payload = http.getString();
  http.end();

  if (code != 200) {
    Serial.print("Auth: signUp gagal ("); Serial.print(code); Serial.println(")");
    Serial.println(payload);
    return false;
  }

  StaticJsonDocument<1024> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) {
    Serial.println(String("Auth: JSON parse error: ") + err.c_str());
    return false;
  }

  g_idToken      = doc["idToken"]      | "";
  g_refreshToken = doc["refreshToken"] | "";
  const char* expStr = doc["expiresIn"] | "3600";

  unsigned long expSec = strtoul(expStr, nullptr, 10);
  // refresh 60 detik lebih awal
  g_tokenExpiryMs = millis() + (expSec > 60 ? (expSec - 60) : expSec) * 1000UL;

  bool ok = (g_idToken.length() > 0);
  Serial.println(ok ? "✅ Auth: Anonymous OK" : "❌ Auth: idToken kosong");
  return ok;
}

bool firebaseRefreshTokenIfNeeded() {
  if (g_idToken.length() == 0) return firebaseSignInAnon();
  if (millis() < g_tokenExpiryMs) return true; // masih valid
  if (g_refreshToken.length() == 0) return firebaseSignInAnon();

  WiFiClientSecure client; client.setInsecure();
  HTTPClient http;

  String url = "https://securetoken.googleapis.com/v1/token?key=" + FIREBASE_WEB_API_KEY;
  if (!http.begin(client, url)) return false;

  http.addHeader("Content-Type", "application/x-www-form-urlencoded");
  String body = "grant_type=refresh_token&refresh_token=" + g_refreshToken;

  int code = http.POST(body);
  String payload = http.getString();
  http.end();

  if (code != 200) {
    Serial.print("Auth: refresh gagal ("); Serial.print(code); Serial.println(")");
    Serial.println(payload);
    // coba login ulang
    return firebaseSignInAnon();
  }

  StaticJsonDocument<1024> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) return false;

  g_idToken      = doc["id_token"]      | g_idToken;
  g_refreshToken = doc["refresh_token"] | g_refreshToken;

  const char* expStr = doc["expires_in"] | "3600";
  unsigned long expSec = strtoul(expStr, nullptr, 10);
  g_tokenExpiryMs = millis() + (expSec > 60 ? (expSec - 60) : expSec) * 1000UL;

  Serial.println("🔁 Auth: token diperbarui");
  return true;
}

// ================= FIREBASE TEST =================
void setupFirebase() {
  lcd.setCursor(0, 3); lcd.print("Tes Firebase... ");
  Serial.println("Test Firebase...");

  // Pastikan sudah punya idToken
  if (!firebaseRefreshTokenIfNeeded()) {
    firebaseSiap = false;
    Serial.println("❌ Auth gagal");
    lcd.setCursor(0, 3); lcd.print("Firebase: Auth NG");
    return;
  }

  WiFiClientSecure client; client.setInsecure();
  HTTPClient http;

  String testUrl = withAuth("/tes_koneksi.json"); // https://host/tes_koneksi.json?auth=ID_TOKEN
  if (!http.begin(client, testUrl)) {
    Serial.println("❌ http.begin gagal");
    firebaseSiap = false;
    lcd.setCursor(0, 3); lcd.print("Firebase: Gagal ");
    return;
  }

  http.addHeader("Content-Type", "application/json");
  String testData = "\"tes_" + String(millis()) + "\"";
  int kodeResponse = http.PUT(testData);

  if (kodeResponse > 0 && (kodeResponse == 200 || kodeResponse == 204)) {
    firebaseSiap = true;
    Serial.println("✅ Firebase OK");
    lcd.setCursor(0, 3); lcd.print("Firebase: OK ");
  } else {
    firebaseSiap = false;
    Serial.println("❌ Firebase gagal, kode: " + String(kodeResponse));
    lcd.setCursor(0, 3); lcd.print("Firebase: Gagal ");
  }
  http.end();

  // bersihkan key uji
  if (firebaseSiap) {
    if (http.begin(client, testUrl)) {
      http.sendRequest("DELETE");
      http.end();
    }
  }
}

// ================= SENSOR INIT =================
void setupSensorSempurna() {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Siapkan sensor...");

  bool mlxBerhasil = false;
  for (int coba = 0; coba < 3; coba++) {
    if (mlx.begin()) { mlxBerhasil = true; break; }
    delay(200);
  }

  if (mlxBerhasil) {
    Serial.println("✅ MLX90614 OK");
    lcd.setCursor(0, 1); lcd.print("Sensor suhu: OK ");
  } else {
    Serial.println("⚠ MLX90614 fallback");
    lcd.setCursor(0, 1); lcd.print("Suhu: Fallback ");
  }

  bool maxBerhasil = false;
  for (int coba = 0; coba < 3; coba++) {
    if (sensorDetak.begin(Wire, I2C_SPEED_STANDARD)) { maxBerhasil = true; break; }
    delay(200);
  }

  if (maxBerhasil) {
    sensorDetak.setup();
    sensorDetak.setPulseAmplitudeRed(0x3F);
    sensorDetak.setPulseAmplitudeIR(0x3F);
    Serial.println("✅ MAX30105 OK");
    lcd.setCursor(0, 2); lcd.print("Detak/SpO2: OK ");
  } else {
    Serial.println("⚠ MAX30105 fallback");
    lcd.setCursor(0, 2); lcd.print("Detak: Fallback ");
  }

  lcd.setCursor(0, 3); lcd.print("Semua: Siap ");
  delay(500);
}

// ================= STATUS =================
void tampilkanStatus() {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Monitor Siap ");
  lcd.setCursor(0, 1); lcd.print("WiFi: "); lcd.print(wifiTerhubung ? "OK " : "Offline");
  lcd.setCursor(0, 2); lcd.print("NTP: ");  lcd.print(ntpSinkron ? "Sync " : "Manual ");
  lcd.setCursor(0, 3); lcd.print("Tekan tombol biru ");
}

// ================= SUHU (15 detik, TANPA RATA-RATA, hanya SUHU OBJEK) =================
void ukurSuhuTubuh() {
  unsigned long tMulai = millis();

  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Ukur suhu 15s   ");

  // kita simpan nilai terakhir yang VALID saja (tanpa averaging)
  float lastValidObj = NAN;
  unsigned long lastPrint = 0;

  Serial.println("=== SUHU TUBUH (Objek saja, 15s, TANPA rata-rata) ===");

  while (millis() - tMulai < 15000UL) {   // 15 detik
    float sObj = mlx.readObjectTempC();   // suhu objek (tubuh)

    // Validasi sederhana (anti-NaN & rentang manusiawi untuk termometer IR)
    // kamu bisa sesuaikan rentang jika perlu (30–43 C umumnya untuk pengukuran dahi/permukaan kulit)
    if (!isnan(sObj) && sObj > 30.0f && sObj < 43.5f) {
      lastValidObj = sObj;                // AMBIL NILAI TERAKHIR YANG VALID (bukan rata-rata)
      suhuTubuh    = lastValidObj;        // update nilai global realtime
    }
    // kalau tidak valid, suhuTubuh tetap mempertahankan nilai valid terakhir (atau fallback jika belum ada)

    // Tampil tiap 1 detik ke LCD & Serial
    if (millis() - lastPrint >= 1000UL) {
      lastPrint = millis();

      float tampil = isnan(lastValidObj) ? suhuFallback : lastValidObj;

      lcd.setCursor(0, 1); lcd.print("Obj: " + String(tampil, 1) + "C     ");
      lcd.setCursor(0, 2); lcd.print("T:" + String((millis() - tMulai) / 1000) + "/15s     ");
      lcd.setCursor(0, 3); lcd.print(isnan(lastValidObj) ? "Status: Estim " : "Status: OK    ");

      Serial.print("Suhu Objek (live): ");
      Serial.print(tampil, 2);
      Serial.println(" *C");
    }

    delay(40); // ~25 Hz sampling
  }

  // Jika dari awal tidak ada sampel valid, pakai fallback
  if (isnan(lastValidObj)) {
    suhuTubuh = suhuFallback; // default 36.5
    Serial.println("⚠ Fallback suhu objek: " + String(suhuTubuh, 1) + "C (tak ada sampel valid)");
  } else {
    // pastikan hasil akhir = nilai valid TERAKHIR (bukan rata-rata)
    suhuTubuh = lastValidObj;
  }

  Serial.println("=== HASIL SUHU (tanpa rata-rata) ===");
  Serial.println("Objek (last): " + String(suhuTubuh, 2) + " *C");
  Serial.println("====================================");
  delay(300);
}

// ================= BPM (30s) =================
void ukurDetakJantungFixed() {
  int detik = 0;
  unsigned long start = millis();
  unsigned long lastPrint = millis();

  bpm_final = 0;
  lastBeat  = 0;

  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Deteksi BPM: 30s ");
  delay(300);

  Serial.println("=== BPM (30s) ===");

  while (millis() - start < durasi_ms) {
    sensorDetak.check();
    long irValue = sensorDetak.getIR();

    if (checkForBeat(irValue)) {
      long delta = millis() - lastBeat;
      lastBeat = millis();

      if (delta > 0) {
        beatsPerMinute = 60.0 / (delta / 1000.0);
        if (beatsPerMinute >= 40 && beatsPerMinute <= 180) {
          bpm_final = (int)beatsPerMinute;
        }
      }
    }

    if (millis() - lastPrint >= 1000 && detik < 30) {
      detik++;
      lastPrint += 1000;
      lcd.clear();
      lcd.setCursor(0, 0); lcd.print("BPM: " + String(bpm_final));
      lcd.setCursor(0, 1); lcd.print("T: " + String(detik) + "/30 ");
      Serial.print("Detik "); Serial.print(detik);
      Serial.print(" => BPM: "); Serial.println(bpm_final);
    }

    delay(10);
  }

  if (bpm_final == 0) bpm_final = detakFallback;
  detakJantungFinal = bpm_final;
  Serial.println("✅ Hasil BPM: " + String(detakJantungFinal));
}

// ================= SpO2 (30s) =================
void ukurKadarOksigen() {
  int detik = 0;

  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Deteksi SpO2:30s ");
  delay(250);

  Serial.println("=== SpO2 (30s) ===");

  for (int k = 0; k < UKURAN_BUFFER; k++) {
    sensorDetak.check();
    if (sensorDetak.available()) {
      bufferMerah[k] = sensorDetak.getRed();
      bufferIR[k]    = sensorDetak.getIR();
      sensorDetak.nextSample();
    } else {
      bufferMerah[k] = 100000 + random(-10000, 10000);
      bufferIR[k]    =  50000 + random( -5000,  5000);
    }
    delay(4);
  }

  unsigned long mulai = millis();
  unsigned long lastPrint = millis();

  while (millis() - mulai < durasi_ms) {
    for (int k = 0; k < UKURAN_BUFFER - 25; k++) {
      bufferMerah[k] = bufferMerah[k + 25];
      bufferIR[k]    = bufferIR[k + 25];
    }

    for (int k = UKURAN_BUFFER - 25; k < UKURAN_BUFFER; k++) {
      sensorDetak.check();
      if (sensorDetak.available()) {
        bufferMerah[k] = sensorDetak.getRed();
        bufferIR[k]    = sensorDetak.getIR();
        sensorDetak.nextSample();
      } else {
        bufferMerah[k] = 100000 + random(-10000, 10000);
        bufferIR[k]    =  50000 + random( -5000,  5000);
      }
      delay(2);
    }

    maxim_heart_rate_and_oxygen_saturation(
      bufferIR, UKURAN_BUFFER, bufferMerah,
      &spo2, &spo2Valid, &heartRate, &heartRateValid
    );

    if (spo2Valid == 1 && spo2 >= 70 && spo2 <= 100) {
      spo2Final = spo2;
    }

    if (millis() - lastPrint >= 1000 && detik < 30) {
      detik++;
      lastPrint += 1000;
      lcd.clear();
      lcd.setCursor(0, 0); lcd.print("SpO2: " + String(spo2Final) + "% ");
      lcd.setCursor(0, 1); lcd.print("T: " + String(detik) + "/30 ");
      Serial.print("Detik "); Serial.print(detik);
      Serial.print(" => SpO2: "); Serial.println(spo2Final);
    }

    delay(10);
  }

  if (spo2Final == 0) spo2Final = spo2Fallback;
  Serial.println("✅ Hasil SpO2: " + String(spo2Final) + "%");
}

// ================= TENSI (blocking, tanpa estimasi) =================
void ukurTekananDarahFixed() {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Cek Tekanan Darah");
  Serial.println("=== TEKANAN DARAH ===");

  b_read = 0; b_discard = 0; i = 0; j = 0;

  const char spinner[4] = { '|', '/', '-', '\\' };
  int  spin = 0;
  const char HEADER[] = { 'e', 'r', 'r', ':', '0' };
  int  matchIdx = 0;

  lcd.setCursor(0, 1); lcd.print("Menunggu alat...");

  while (b_read == 0) {
    while (Serial2.available()) {
      char c = (char)Serial2.read();
      if (c == HEADER[matchIdx]) {
        matchIdx++;
        if (matchIdx == 5) { b_read = 1; break; }
      } else {
        matchIdx = (c == HEADER[0]) ? 1 : 0;
      }
    }
    lcd.setCursor(15, 1); lcd.print(spinner[(spin++) & 3]);
    delay(40);
  }

  lcd.setCursor(0, 1); lcd.print("Header: OK ");

  while (i < 30) {
    if (Serial2.available()) {
      discard = Serial2.read();
      i++;
      if ((i % 5) == 0) {
        lcd.setCursor(0, 2);
        lcd.print("Buang: "); lcd.print(i); lcd.print("/30 ");
      }
    } else {
      delay(2);
    }
  }

  lcd.setCursor(0, 2); lcd.print("Ambil data... ");

  while (j < 11) {
    if (Serial2.available()) {
      final_buff[j] = Serial2.read();
      j++;
      lcd.setCursor(12, 2); lcd.print(j); lcd.print("/11");
    } else {
      delay(2);
    }
  }

  int hexSys  = hexToDec(final_buff[0], final_buff[1]);
  int hexDias = hexToDec(final_buff[3], final_buff[4]);
  int hexBPM  = hexToDec(final_buff[9], final_buff[10]);

  nilaiSistolik   = hexSys  + KALIBRASI_SISTOLIK;
  nilaiDiastolik  = hexDias + KALIBRASI_DIASTOLIK;
  nilaiDetakDarah = hexBPM;

  Serial.println(
    "SYS/DIA/BPM (kalibrasi): " +
    String(nilaiSistolik) + "/" + String(nilaiDiastolik) +
    " / " + String(nilaiDetakDarah)
  );

  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("SYS: " + String(nilaiSistolik) + " ");
  lcd.setCursor(0, 1); lcd.print("DIA: " + String(nilaiDiastolik) + " ");
  lcd.setCursor(0, 2); lcd.print("BPM: " + String(nilaiDetakDarah) + " ");
}

// ================= HASIL AKHIR =================
void tampilkanHasil() {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("=== HASIL AKHIR ===");
  lcd.setCursor(0, 1); lcd.print("Suhu: " + String(suhuTubuh, 1) + "C ");
  lcd.setCursor(0, 2); lcd.print("BPM : " + String(detakJantungFinal) + " ");
  lcd.setCursor(0, 3); lcd.print("SpO2: " + String(spo2Final) + "% ");
  delay(1000);

  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("SYS: " + String(nilaiSistolik)  + " mmHg ");
  lcd.setCursor(0, 1); lcd.print("DIA: " + String(nilaiDiastolik) + " mmHg");
  lcd.setCursor(0, 2); lcd.print("NB : " + nbLabel + " ");
  lcd.setCursor(0, 3); lcd.print("Rule: " + getStatusKesehatan());

  Serial.println("\n=== RINGKASAN ===");
  Serial.println("Suhu: " + String(suhuTubuh, 1) + "C");
  Serial.println("BPM : " + String(detakJantungFinal));
  Serial.println("SpO2: " + String(spo2Final) + "%");
  Serial.println("BP : "  + String(nilaiSistolik) + "/" + String(nilaiDiastolik) + " mmHg");
  Serial.println("NB : "  + nbLabel);
  Serial.println("Rule: " + getStatusKesehatan());
  Serial.println("==================");

  delay(1000);
}

// ================= RULE-BASED (range suhu 36–38) =================
String getStatusKesehatan() {
  int skor = 0;

  if (suhuTubuh >= 36.0 && suhuTubuh <= 38.0)        skor += 25;
  else if (suhuTubuh >= 35.0 && suhuTubuh <= 38.5)   skor += 15;

  if (detakJantungFinal >= 60 && detakJantungFinal <= 100)       skor += 25;
  else if (detakJantungFinal >= 50 && detakJantungFinal <= 120)  skor += 15;

  if (spo2Final >= 96) skor += 25;
  else if (spo2Final >= 90) skor += 15;

  if (nilaiSistolik >= 90 && nilaiDiastolik >= 60 &&
      nilaiSistolik <= 130 && nilaiDiastolik <= 85) {
    skor += 25;
  } else if (nilaiSistolik >= 80 && nilaiDiastolik >= 50 &&
             nilaiSistolik <= 150 && nilaiDiastolik <= 95) {
    skor += 15;
  }

  if (skor >= 85) return "Normal";
  if (skor >= 60) return "Kurang Normal";
  return "Berbahaya";
}

// ================= KIRIM KE FIREBASE =================
void kirimDataKeFirebaseFixed() {
  if (!firebaseRefreshTokenIfNeeded()) {
    Serial.println("❌ Auth tak tersedia saat kirim");
    return;
  }

  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Kirim ke server...");
  Serial.println("\n--- Kirim ke Firebase ---");

  unsigned long waktuStampProper = ntpSinkron ? getNTPTime() : millis();

  StaticJsonDocument<896> doc;
  doc["waktu"]            = waktuStampProper;
  doc["perangkat"]        = DEVICE_ID;
  doc["suhu_tubuh"]       = String(suhuTubuh, 1);
  doc["detak_jantung"]    = String(detakJantungFinal);
  doc["kadar_oksigen"]    = String(spo2Final);
  doc["tekanan_sistolik"] = String(nilaiSistolik);
  doc["tekanan_diastolik"]= String(nilaiDiastolik);
  doc["status_kesehatan"] = getStatusKesehatan();
  doc["prediksi_nb"]      = nbLabel;
  doc["waktu_baca"]       = getNTPTimestamp();

  String jsonString;
  serializeJson(doc, jsonString);

  String pathData   = "/data_kesehatan/data_" + String(waktuStampProper) + ".json";
  String urlData    = withAuth(pathData);
  String urlTerbaru = withAuth("/data_kesehatan/terbaru.json");

  WiFiClientSecure client; client.setInsecure();
  HTTPClient http;

  if (http.begin(client, urlData)) {
    http.addHeader("Content-Type", "application/json");
    int kode = http.PUT(jsonString);

    lcd.setCursor(0, 1);
    if (kode > 0 && (kode == 200 || kode == 204)) {
      Serial.println("✅ Simpan OK (" + String(kode) + ")");
      lcd.print("Berhasil dikirim ");
    } else {
      Serial.println("❌ Simpan gagal (" + String(kode) + ")");
      lcd.print("Gagal kirim (" + String(kode) + ") ");
    }
    http.end();
  }

  if (http.begin(client, urlTerbaru)) {
    http.addHeader("Content-Type", "application/json");
    int kode2 = http.PUT(jsonString);

    lcd.setCursor(0, 2);
    if (kode2 > 0 && (kode2 == 200 || kode2 == 204)) {
      Serial.println("✅ Terbaru OK");
      lcd.print("Real-time: OK ");
    } else {
      Serial.println("⚠ Terbaru gagal (" + String(kode2) + ")");
      lcd.print("Real-time: Gagal ");
    }
    http.end();
  }

  Serial.println("JSON: " + jsonString);
  delay(700);
}

// ================= NTP UTILS =================
unsigned long getNTPTime() {
  time_t now;
  time(&now);
  return (unsigned long)now * 1000UL; // ms
}

String getNTPTimestamp() {
  if (!ntpSinkron) {
    unsigned long totalDetik = millis() / 1000;
    int jam   = (totalDetik / 3600) % 24;
    int menit = (totalDetik / 60) % 60;
    int detik =  totalDetik % 60;

    char waktuStr[10];
    sprintf(waktuStr, "%02d:%02d:%02d", jam, menit, detik);
    return String(waktuStr);
  }

  struct tm timeinfo;
  if (!getLocalTime(&timeinfo)) return "NTP Error";

  char buf[32];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &timeinfo);
  return String(buf);
}

// ================= NB HELPERS =================
void standardizeFeatures(
  float z[F_COUNT],
  float rawTemp, int rawHR, int rawSYS, int rawDIA, int rawSpO2
) {
  float raw[F_COUNT] = {
    rawTemp, (float)rawHR, (float)rawSYS, (float)rawDIA, (float)rawSpO2
  };

  for (int k = 0; k < F_COUNT; k++) {
    float s = stdz_std[k];
    if (s < 1e-6f) s = 1.0f;
    z[k] = (raw[k] - stdz_mean[k]) / s;
  }
}

int argmax3(float a, float b, float c) {
  if (a >= b && a >= c) return 0;
  if (b >= a && b >= c) return 1;
  return 2;
}

String predictNB_fromZ(const float z[F_COUNT]) {
  const float LOG2PI = 1.8378770664093453f; // ln(2*pi)
  float logpost[NB_NUM_CLASSES] = { 0, 0, 0 };

  for (int c = 0; c < NB_NUM_CLASSES; c++) {
    float prior = nbPrior[c];
    if (prior < 1e-12f) prior = 1e-12f;

    float lp = log(prior);

    for (int f = 0; f < F_COUNT; f++) {
      float mu = nbMu[c][f];
      float s  = nbSigma[c][f];
      if (s < 1e-4f) s = 1e-4f;

      float diff = z[f] - mu;
      lp += -0.5f * (LOG2PI + 2.0f * log(s) + (diff * diff) / (s * s));
    }
    logpost[c] = lp;
  }

  int winner = argmax3(logpost[0], logpost[1], logpost[2]);
  return NB_LABELS[winner];
}

String predictNB(float rawTemp, int rawHR, int rawSYS, int rawDIA, int rawSpO2) {
  float z[F_COUNT];
  standardizeFeatures(z, rawTemp, rawHR, rawSYS, rawDIA, rawSpO2);
  return predictNB_fromZ(z);
}

// Load model NB dari Firebase: /model/naive_bayes_model.json
bool loadNBModelFromFirebase() {
  if (!firebaseRefreshTokenIfNeeded()) return false;

  WiFiClientSecure client; client.setInsecure();
  HTTPClient http;

  String url = withAuth("/model/naive_bayes_model.json");
  if (!http.begin(client, url)) {
    Serial.println("NB: http.begin gagal");
    return false;
  }

  int code = http.GET();
  if (code != 200) {
    Serial.println("NB: GET gagal (" + String(code) + ")");
    http.end();
    return false;
  }

  String payload = http.getString();
  http.end();

  StaticJsonDocument<8192> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) {
    Serial.println(String("NB: JSON parse error: ") + err.c_str());
    return false;
  }

  bool okPrior = false, okMeans = false, okStd = false;

  if (doc.containsKey("prior")) {
    JsonObject pr = doc["prior"].as<JsonObject>();
    nbPrior[0] = pr["0"] | nbPrior[0];
    nbPrior[1] = pr["1"] | nbPrior[1];
    nbPrior[2] = pr["2"] | nbPrior[2];
    okPrior = true;
  } else {
    Serial.println("NB: prior tidak ditemukan");
  }

  if (doc.containsKey("means")) {
    JsonObject ms = doc["means"].as<JsonObject>();
    JsonObject m0 = ms["0"].as<JsonObject>();
    JsonObject m1 = ms["1"].as<JsonObject>();
    JsonObject m2 = ms["2"].as<JsonObject>();

    const char* K_TEMP = "Suhu Tubuh (C)";
    const char* K_HR   = "Detak Jantung";
    const char* K_SYS  = "Sistolik";
    const char* K_DIA  = "Diastolik";
    const char* K_SPO2 = "Saturasi Oksigen";

    nbMu[0][F_TEMP] = m0[K_TEMP] | nbMu[0][F_TEMP];
    nbMu[0][F_HR ]  = m0[K_HR ]  | nbMu[0][F_HR ];
    nbMu[0][F_SYS ] = m0[K_SYS ] | nbMu[0][F_SYS ];
    nbMu[0][F_DIA ] = m0[K_DIA ] | nbMu[0][F_DIA ];
    nbMu[0][F_SPO2] = m0[K_SPO2] | nbMu[0][F_SPO2];

    nbMu[1][F_TEMP] = m1[K_TEMP] | nbMu[1][F_TEMP];
    nbMu[1][F_HR ]  = m1[K_HR ]  | nbMu[1][F_HR ];
    nbMu[1][F_SYS ] = m1[K_SYS ] | nbMu[1][F_SYS ];
    nbMu[1][F_DIA ] = m1[K_DIA ] | nbMu[1][F_DIA ];
    nbMu[1][F_SPO2] = m1[K_SPO2] | nbMu[1][F_SPO2];

    nbMu[2][F_TEMP] = m2[K_TEMP] | nbMu[2][F_TEMP];
    nbMu[2][F_HR ]  = m2[K_HR ]  | nbMu[2][F_HR ];
    nbMu[2][F_SYS ] = m2[K_SYS ] | nbMu[2][F_SYS ];
    nbMu[2][F_DIA ] = m2[K_DIA ] | nbMu[2][F_DIA ];
    nbMu[2][F_SPO2] = m2[K_SPO2] | nbMu[2][F_SPO2];

    okMeans = true;
  } else {
    Serial.println("NB: means tidak ditemukan");
  }

  if (doc.containsKey("std")) {
    JsonObject sd = doc["std"].as<JsonObject>();
    JsonObject s0 = sd["0"].as<JsonObject>();
    JsonObject s1 = sd["1"].as<JsonObject>();
    JsonObject s2 = sd["2"].as<JsonObject>();

    const char* K_TEMP = "Suhu Tubuh (C)";
    const char* K_HR   = "Detak Jantung";
    const char* K_SYS  = "Sistolik";
    const char* K_DIA  = "Diastolik";
    const char* K_SPO2 = "Saturasi Oksigen";

    nbSigma[0][F_TEMP] = s0[K_TEMP] | nbSigma[0][F_TEMP];
    nbSigma[0][F_HR ]  = s0[K_HR ]  | nbSigma[0][F_HR ];
    nbSigma[0][F_SYS ] = s0[K_SYS ] | nbSigma[0][F_SYS ];
    nbSigma[0][F_DIA ] = s0[K_DIA ] | nbSigma[0][F_DIA ];
    nbSigma[0][F_SPO2] = s0[K_SPO2] | nbSigma[0][F_SPO2];

    nbSigma[1][F_TEMP] = s1[K_TEMP] | nbSigma[1][F_TEMP];
    nbSigma[1][F_HR ]  = s1[K_HR ]  | nbSigma[1][F_HR ];
    nbSigma[1][F_SYS ] = s1[K_SYS ] | nbSigma[1][F_SYS ];
    nbSigma[1][F_DIA ] = s1[K_DIA ] | nbSigma[1][F_DIA ];
    nbSigma[1][F_SPO2] = s1[K_SPO2] | nbSigma[1][F_SPO2];

    nbSigma[2][F_TEMP] = s2[K_TEMP] | nbSigma[2][F_TEMP];
    nbSigma[2][F_HR ]  = s2[K_HR ]  | nbSigma[2][F_HR ];
    nbSigma[2][F_SYS ] = s2[K_SYS ] | nbSigma[2][F_SYS ];
    nbSigma[2][F_DIA ] = s2[K_DIA ] | nbSigma[2][F_DIA ];
    nbSigma[2][F_SPO2] = s2[K_SPO2] | nbSigma[2][F_SPO2];

    okStd = true;
  } else {
    Serial.println("NB: std tidak ditemukan");
  }

  return (okPrior && okMeans && okStd);
}

// (Opsional) Load standardizer (z-score) dari Firebase: /model/standardizer.json
bool loadStandardizerFromFirebase() {
  if (!firebaseRefreshTokenIfNeeded()) return false;

  WiFiClientSecure client; client.setInsecure();
  HTTPClient http;

  String url = withAuth("/model/standardizer.json");
  if (!http.begin(client, url)) {
    return false;
  }

  int code = http.GET();
  if (code != 200) {
    http.end();
    return false;
  }

  String payload = http.getString();
  http.end();

  StaticJsonDocument<4096> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) {
    return false;
  }

  const char* K_TEMP = "Suhu Tubuh (C)";
  const char* K_HR   = "Detak Jantung";
  const char* K_SYS  = "Sistolik";
  const char* K_DIA  = "Diastolik";
  const char* K_SPO2 = "Saturasi Oksigen";

  if (doc.containsKey("mean")) {
    JsonObject M = doc["mean"].as<JsonObject>();
    stdz_mean[F_TEMP] = M[K_TEMP] | stdz_mean[F_TEMP];
    stdz_mean[F_HR ]  = M[K_HR ]  | stdz_mean[F_HR ];
    stdz_mean[F_SYS ] = M[K_SYS ] | stdz_mean[F_SYS ];
    stdz_mean[F_DIA ] = M[K_DIA ] | stdz_mean[F_DIA ];
    stdz_mean[F_SPO2] = M[K_SPO2] | stdz_mean[F_SPO2];
  }

  if (doc.containsKey("std")) {
    JsonObject S = doc["std"].as<JsonObject>();
    stdz_std[F_TEMP] = S[K_TEMP] | stdz_std[F_TEMP];
    stdz_std[F_HR ]  = S[K_HR ]  | stdz_std[F_HR ];
    stdz_std[F_SYS ] = S[K_SYS ] | stdz_std[F_SYS ];
    stdz_std[F_DIA ] = S[K_DIA ] | stdz_std[F_DIA ];
    stdz_std[F_SPO2] = S[K_SPO2] | stdz_std[F_SPO2];
  }

  return true;
}

// ================= UTIL =================
int hexDigit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return 0;
}

int hexToDec(char high, char low) {
  return (hexDigit(high) << 4) | hexDigit(low);
}

String withAuth(const String& path) {
  // Selalu tambahkan ?auth=<idToken> agar tidak 401 (DB rules default: locked)
  String url = "https://" + FIREBASE_HOST + path;
  if (g_idToken.length() > 0) {
    url += (url.indexOf('?') >= 0 ? "&" : "?");
    url += "auth=" + g_idToken;
  }
  return url;
}