from codecarbon import EmissionsTracker


class Energy:
    tracker = None
    emissions_data = None
    emissions = 0
    def start(self):
        self.tracker = EmissionsTracker(log_level='critical')
        self.tracker.start()

    def end(self):
        self.emissions = self.tracker.stop()
        self.emissions_data = self.tracker.final_emissions_data

    def get_emissions(self):
        return self.emissions

    def get_energy(self):
        return self.emissions_data.energy_consumed

    def print_emissions(self):
        print(f"Emissions: {self.emissions} kg")
