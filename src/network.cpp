#include <vector>
#include <cstdint>
#include <ctime>
#include <cstdlib>

using namespace std;

struct Gate {
    uint8_t truthTable; // 8-bit truth table for 3 inputs (0..7)
    int in[3];          // Wire IDs: [0 .. numIn-1] are primary inputs, [numIn ..] are gate outputs

    inline uint8_t Process(const vector<uint8_t>& wires) const {
        int index = (wires[in[0]] << 2) | (wires[in[1]] << 1) | wires[in[2]];
        return (truthTable >> index) & 1;
    }
};

class Network {
private:
    int numInputs;
    vector<Gate> gates;       // Topologically ordered: a gate only reads from wires < its own wire ID
    vector<int> outputWires;  // Wire IDs that serve as final network outputs

	void CreateNetwork(const vector<vector<uint8_t>>& partAX, const vector<vector<uint8_t>>& partAY) {
		gates.clear();
		outputWires.clear();

		if (partAX.empty() || partAY.empty()) {
			return;
		}

		size_t outputSize = partAY[0].size();
		int zeroConstWire = -1;
		auto getZeroWire = [&]() -> int {
			if (zeroConstWire == -1) {
				gates.push_back(Gate{0x00, {0, 0, 0}});
				zeroConstWire = numInputs + static_cast<int>(gates.size()) - 1;
			}
			return zeroConstWire;
		};

		if (numInputs == 0) {
			for (size_t outIdx = 0; outIdx < outputSize; outIdx++) {
				uint8_t val = 0;
				for (size_t i = 0; i < partAY.size(); i++) {
					if (partAY[i][outIdx]) {
						val = 1;
						break;
					}
				}
				gates.push_back(Gate{static_cast<uint8_t>(val ? 0xFF : 0x00), {0, 0, 0}});
				outputWires.push_back(numInputs + static_cast<int>(gates.size()) - 1);
			}
			return;
		}

		// 1. Build a decoder chain for each sample in partA
		vector<int> sampleChains(partAX.size());

		for (size_t i = 0; i < partAX.size(); i++) {
			const auto& x = partAX[i];
			int currentWire = -1;

			if (numInputs == 1) {
				int idx = (x[0] == 1) ? 7 : 0;
				gates.push_back(Gate{static_cast<uint8_t>(1 << idx), {0, 0, 0}});
				currentWire = numInputs + static_cast<int>(gates.size()) - 1;
			} else if (numInputs == 2) {
				int idx = (x[0] << 2) | (x[1] ? 3 : 0);
				gates.push_back(Gate{static_cast<uint8_t>(1 << idx), {0, 1, 1}});
				currentWire = numInputs + static_cast<int>(gates.size()) - 1;
			} else {
				int idx0 = (x[0] << 2) | (x[1] << 1) | x[2];
				gates.push_back(Gate{static_cast<uint8_t>(1 << idx0), {0, 1, 2}});
				currentWire = numInputs + static_cast<int>(gates.size()) - 1;

				for (int j = 3; j < numInputs; j += 2) {
					int in1 = j;
					int in2 = (j + 1 < numInputs) ? (j + 1) : j;
					uint8_t b1 = x[in1] & 1;
					uint8_t b2 = x[in2] & 1;
					int chainIdx = (1 << 2) | (b1 << 1) | b2;
					gates.push_back(Gate{static_cast<uint8_t>(1 << chainIdx), {currentWire, in1, in2}});
					currentWire = numInputs + static_cast<int>(gates.size()) - 1;
				}
			}

			sampleChains[i] = currentWire;
		}

		// 2. Build 3-to-1 OR tree for each output bit
		const uint8_t OR3_TT = 0xFE;

		for (size_t outIdx = 0; outIdx < outputSize; outIdx++) {
			vector<int> activeWires;
			for (size_t i = 0; i < partAY.size(); i++) {
				if (partAY[i][outIdx] == 1) {
					activeWires.push_back(sampleChains[i]);
				}
			}

			if (activeWires.empty()) {
				outputWires.push_back(getZeroWire());
				continue;
			}

			while (activeWires.size() > 1) {
				vector<int> nextLevel;
				for (size_t k = 0; k < activeWires.size(); k += 3) {
					int w0 = activeWires[k];
					int w1 = (k + 1 < activeWires.size()) ? activeWires[k + 1] : w0;
					int w2 = (k + 2 < activeWires.size()) ? activeWires[k + 2] : w0;
					gates.push_back(Gate{OR3_TT, {w0, w1, w2}});
					nextLevel.push_back(numInputs + static_cast<int>(gates.size()) - 1);
				}
				activeWires = nextLevel;
			}

			outputWires.push_back(activeWires[0]);
		}
	}

public:
    Network(const vector<vector<uint8_t>>& X, const vector<vector<uint8_t>>& Y, double partA = 0.3) {
        numInputs = X.empty() ? 0 : X[0].size();
        vector<vector<uint8_t>> partAX, partAY;

        srand(time(nullptr));
        for (size_t i = 0; i < X.size(); i++) {
            if ((static_cast<double>(rand()) / RAND_MAX) < partA) {
                partAX.push_back(X[i]);
                partAY.push_back(Y[i]);
            }
        }
        CreateNetwork(partAX, partAY);
    }

    vector<uint8_t> Process(const vector<uint8_t>& input) const {
        vector<uint8_t> wires(numInputs + gates.size());

        // 1. Load primary inputs into the first `numInputs` wires
        for (int i = 0; i < numInputs; i++) {
            wires[i] = input[i] & 1;
        }

        // 2. Evaluate all gates in a single linear pass (no recursion, no filledSlots deadlock)
        for (size_t g = 0; g < gates.size(); g++) {
            wires[numInputs + g] = gates[g].Process(wires);
        }

        // 3. Collect output wires
        vector<uint8_t> ret(outputWires.size());
        for (size_t i = 0; i < outputWires.size(); i++) {
            ret[i] = wires[outputWires[i]];
        }
        return ret;
    }

    // Total mismatch count on any dataset split (for Stage 2 optimization selection)
    int Loss(const vector<vector<uint8_t>>& X, const vector<vector<uint8_t>>& Y) const {
        int errors = 0;
        for (size_t s = 0; s < X.size(); s++) {
            vector<uint8_t> pred = Process(X[s]);
            if (pred != Y[s]) {
                errors++;
            }
        }
        return errors;
    }
};
