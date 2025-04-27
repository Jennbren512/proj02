#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Constants
const float GRAIN_GROWS_PER_MONTH = 12.0;
const float ONE_DEER_EATS_PER_MONTH = 1.0;

const float AVG_PRECIP_PER_MONTH = 7.0;
const float AMP_PRECIP_PER_MONTH = 6.0;
const float RANDOM_PRECIP = 2.0;

const float AVG_TEMP = 60.0;
const float AMP_TEMP = 20.0;
const float RANDOM_TEMP = 10.0;

const float MIDTEMP = 40.0;
const float MIDPRECIP = 10.0;

// Global State Variables
int NowYear = 2025;         // 2025 - 2030
int NowMonth = 0;           // 0 - 11
float NowPrecip;            // inches of rain
float NowTemp;              // temperature (F)
float NowHeight = 5.0;      // grain height (inches)
int NowNumDeer = 2;         // number of deer
int NowNumBears = 0;        // number of bears (our custom agent)

unsigned int seed = 0;

// Barrier Variables
omp_lock_t Lock;
volatile int NumInThreadTeam;
volatile int NumAtBarrier;
volatile int NumGone;

// Function Prototypes
void InitBarrier(int);
void WaitBarrier();
float Ranf(float, float);
int Ranf(int, int);
float SQR(float x) { return x * x; }

void Deer();
void Grain();
void Watcher();
void Bear();

// Main Program
int main() {
    omp_set_num_threads(4);   // 4 sections: Deer, Grain, Watcher, Bear
    InitBarrier(4);

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            Deer();
        }

        #pragma omp section
        {
            Grain();
        }

        #pragma omp section
        {
            Watcher();
        }

        #pragma omp section
        {
            Bear();
        }
    }

    return 0;
}

// Functions

void Deer() {
    while (NowYear < 2031) {
        int nextNumDeer = NowNumDeer;
        int carryingCapacity = (int)(NowHeight);

        if (nextNumDeer < carryingCapacity)
            nextNumDeer++;
        else if (nextNumDeer > carryingCapacity)
            nextNumDeer--;

        if (nextNumDeer < 0) nextNumDeer = 0;

        WaitBarrier();
        NowNumDeer = nextNumDeer;
        WaitBarrier();
        WaitBarrier();
    }
}

void Grain() {
    while (NowYear < 2031) {
        float tempFactor = exp(-SQR((NowTemp - MIDTEMP) / 10.0));
        float precipFactor = exp(-SQR((NowPrecip - MIDPRECIP) / 10.0));

        float nextHeight = NowHeight;
        nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;

        if (nextHeight < 0.) nextHeight = 0.;

        WaitBarrier();
        NowHeight = nextHeight;
        WaitBarrier();
        WaitBarrier();
    }
}

void Watcher() {
    FILE* fp = fopen("simulation_output.csv", "w");

    // Write CSV header
    fprintf(fp, "Year,Month,Temp_C,Precip_cm,Grain_cm,NumDeer,NumBears\n");

    while (NowYear < 2031) {
        WaitBarrier();
        WaitBarrier();

        // Output current state (convert to Celsius and centimeters)
        float tempC = (5.0 / 9.0) * (NowTemp - 32.0);
        float precipCM = NowPrecip * 2.54;
        float heightCM = NowHeight * 2.54;

        fprintf(fp, "%d,%d,%.2f,%.2f,%.2f,%d,%d\n",
                NowYear, NowMonth + 1, tempC, precipCM, heightCM, NowNumDeer, NowNumBears);

        // Advance time
        NowMonth++;
        if (NowMonth > 11) {
            NowMonth = 0;
            NowYear++;
        }

        // Update temperature and precipitation
        float ang = (30. * (float)NowMonth + 15.) * (M_PI / 180.);
        float temp = AVG_TEMP - AMP_TEMP * cos(ang);
        NowTemp = temp + Ranf(-RANDOM_TEMP, RANDOM_TEMP);

        float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
        NowPrecip = precip + Ranf(-RANDOM_PRECIP, RANDOM_PRECIP);
        if (NowPrecip < 0.) NowPrecip = 0.;

        WaitBarrier();
    }

    fclose(fp);
}

void Bear() {
    while (NowYear < 2031) {
        int nextBears = NowNumBears;
        int nextDeer = NowNumDeer;

        if (NowNumDeer > 3) {
            nextBears = 1;
            nextDeer -= 1;
        } else {
            nextBears = 0;
        }

        if (nextDeer < 0) nextDeer = 0;

        WaitBarrier();
        NowNumBears = nextBears;
        NowNumDeer = nextDeer;
        WaitBarrier();
        WaitBarrier();
    }
}

// Barrier Functions

void InitBarrier(int n) {
    NumInThreadTeam = n;
    NumAtBarrier = 0;
    omp_init_lock(&Lock);
}

void WaitBarrier() {
    omp_set_lock(&Lock);
    {
        NumAtBarrier++;
        if (NumAtBarrier == NumInThreadTeam) {
            NumGone = 0;
            NumAtBarrier = 0;
            while (NumGone != NumInThreadTeam - 1);
            omp_unset_lock(&Lock);
            return;
        }
    }
    omp_unset_lock(&Lock);
    while (NumAtBarrier != 0);
    #pragma omp atomic
    NumGone++;
}

// Random Number Functions

float Ranf(float low, float high) {
    float r = (float) rand_r(&seed); // uniform random number between 0 and RAND_MAX
    return (low + r * (high - low) / (float)RAND_MAX);
}

int Ranf(int ilow, int ihigh) {
    float low = (float)ilow;
    float high = (float)ihigh + 0.9999f;
    return (int)(Ranf(low, high));
}
