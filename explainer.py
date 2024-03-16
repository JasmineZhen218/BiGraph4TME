import numpy as np


class Explainer:
    def __init__(self, threshold_hodges_lehmann=0.5):
        self.threshold_hodges_lehmann = threshold_hodges_lehmann

    def hodges_lehmann_estimator(self, x, y):
        """
        Compute the Hodges-Lehmann estimator for the median difference between two samples.

        Parameters:
        - x: numpy array, first sample
        - y: numpy array, second sample

        Returns:
        - Hodges-Lehmann estimator
        """
        n = len(x)
        m = len(y)
        hl_estimates = []

        for i in range(n):
            for j in range(m):
                hl_estimates.append((x[i] - y[j]))

        return np.median(hl_estimates)

    def find_characteristic_patterns(self, Patient_ids, Patient_subgroups, Histograms):
        """
        Find characteristic patterns for each patient subgroup.
        Parameters:
        - Patient_ids: list, patient ids
        - Patient_subgroups: dict, patient subgroups with the key as the subgroup id and the value as a list of patient ids
        - Histograms: list, histograms of characteristic patterns 
        Returns:
        - Characteristic_patterns: dict, characteristic patterns for each patient subgroup. The key is the subgroup id, and the value is a list of pattern ids.

        """
        Histograms = np.stack(Histograms, axis=0) # (n_patients, n_patterns)
        Proportions = Histograms / np.sum(Histograms, axis=1, keepdims=True) # (n_patients, n_patterns) normalized by the sum of each row
        for i in range(len(Patient_subgroups)): # for each patient subgroup
            Patient_ids_in_group = Patient_subgroups[i]['patient_ids'] # patient ids in the subgroup
            Hodges_lehmann = [] # Hodges-Lehmann estimator for each pattern
            for pattern_id in range(Proportions.shape[1]): # for each pattern
                proportion_in_group = Proportions[
                    np.where(np.isin(Patient_ids, Patient_ids_in_group))[0], pattern_id
                ] # proportion of the pattern in the group
                proportion_out_group = Proportions[
                    np.where(~np.isin(Patient_ids, Patient_ids_in_group))[0], pattern_id
                ] # proportion of the pattern out of the group
                hodges_lehmann = self.hodges_lehmann_estimator(
                    proportion_in_group, proportion_out_group
                ) # Hodges-Lehmann estimator for the pattern
                Hodges_lehmann.append(hodges_lehmann) # append the estimator to the list
            
            characteristic_patterns = np.where(
                Hodges_lehmann > self.threshold_hodges_lehmann * np.max(Hodges_lehmann)
            )[0].tolist()
            Patient_subgroups[i]['characteristic_patterns'] = characteristic_patterns
        return Patient_subgroups
