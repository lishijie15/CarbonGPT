import math
import os
import pathlib

import pandas as pd
import py_dss_interface

dss = py_dss_interface.DSSDLL()


class SmartInverterFunction:

    def __init__(self, dss, bus, kV_base, kW_rated, kVA, smart_inverter_function):

        self.dss = dss
        self.bus = bus
        self.kV_base = kV_base
        self.kW_rated = kW_rated
        self.kVA = kVA
        self.smart_inverter_function = smart_inverter_function

        self.__define_3ph_pvsystem_with_transformer()

        self.__define_smart_inverter_function()

    def __define_3ph_pvsystem_with_transformer(self):
        self.__define_transformer()

        self.dss.text("makebuslist")
        self.dss.text(f"setkVBase bus=PV_{self.bus} kVLL=0.48")

        self.__define_pvsystem_curves()

        self.__define_pvsystem()

    def __define_smart_inverter_function(self):
        if self.smart_inverter_function == "unity-pf":
            pass
        elif self.smart_inverter_function == "pf":
            self.__set_pf()
        elif self.smart_inverter_function == "volt-var":
            self.__set_vv()

    def __define_transformer(self):
        self.dss.text(
            f"New transformer.PV_{self.bus} "
            f"phases=3 "
            f"windings=2 "
            f"buses=({self.bus}, PV_{self.bus}) "
            f"conns=(wye, wye) "
            f"kVs=({self.kV_base}, 0.48) "
            f"xhl=5.67 "
            f"%R=0.4726 "
            f"kVAs=({self.kVA}, {self.kVA})")

    def __define_pvsystem_curves(self):
        self.dss.text("New XYCurve.MyPvsT "
                      "npts=4  "
                      "xarray=[0  25  75  100]  "
                      "yarray=[1.0 1.0 1.0 1.0]")

        self.dss.text("New XYCurve.MyEff "
                      "npts=4  "
                      "xarray=[.1  .2  .4  1.0]  "
                      "yarray=[1.0 1.0 1.0 1.0]")

        self.dss.text(
            "New Loadshape.MyIrrad "
            "npts=24 "
            "interval=1 "
            "mult=[0 0 0 0 0 0 .1 .2 .3  .5  .8  .9  1.0  1.0  .99  .9  .7  .4  .1 0  0  0  0  0]")

        self.dss.text(
            "New Tshape.MyTemp "
            "npts=24 "
            "interval=1 "
            "CarbonGPT=[25, 25, 25, 25, 25, 25, 25, 25, 35, 40, 45, 50  60 60  55 40  35  30  25 25 25 25 25 25]")

    def __define_pvsystem(self):
        self.dss.text(
            f"New PVSystem.PV_{self.bus} "
            f"phases=3 "
            f"conn=wye  "
            f"bus1=PV_{self.bus} "
            f"kV=0.48 "
            f"kVA={self.kVA} "
            f"Pmpp={self.kW_rated} "
            f"pf=1 "
            f"%cutin=0.00005 "
            f"%cutout=0.00005 "
            f"VarFollowInverter=yes "
            f"effcurve=Myeff  "
            f"P-TCurve=MyPvsT "
            f"Daily=MyIrrad  "
            f"TDaily=MyTemp "
            f"wattpriority=False")

    def __set_pf(self):
        self.dss.text(f"edit PVSystem.PV_{self.bus} pf=-0.90 pfpriority=True")

    def __set_vv(self):
        x_vv_curve = "[0.5 0.92 0.98 1.0 1.02 1.08 1.5]"
        y_vv_curve = "[1 1 0 0 0 -1 -1]"
        self.dss.text(f"new XYcurve.volt-var npts=7 yarray={y_vv_curve} xarray={x_vv_curve}")
        self.dss.text(
            "new invcontrol.inv "
            "mode=voltvar "
            "voltage_curvex_ref=rated "
            "vvc_curve1=volt-var "
            "RefReactivePower=VARMAX")


def strfill(src, lg, str1):
    n = math.ceil((lg - len(src)) / len(str1))
    newstr = src + str1 * n
    return newstr[0:lg]


# Set the parameters relevant to the power flow calculation.
def set_baseline():
    dss.text("New Energymeter.m1 Line.ln5815900-1 1")
    dss.text("Set Maxiterations=100")
    dss.text("set maxcontrolit=100")
    dss.text("set Maxcontroliter=100")
    dss.text("set Maxiter=100")
    dss.text("Batchedit Load..* daily=default")


def set_time_series_simulation():
    dss.text("set controlmode=Static")
    dss.text("set mode=Snap")
    # dss.text("set number=24")
    # dss.text("set stepsize=1h")


def get_energymeter_results():
    dss.meters_write_name("m1")
    feeder_kwh = dss.meters_register_values()[0]
    feeder_kvarh = dss.meters_register_values()[1]
    loads_kwh = dss.meters_register_values()[4]
    losses_kwh = dss.meters_register_values()[12]
    pv_kwh = loads_kwh + losses_kwh - feeder_kwh

    return feeder_kwh, feeder_kvarh, loads_kwh, losses_kwh, pv_kwh


def powerFlowCalculation():
    smart_inverter_functions_list = ["unity-pf", "pf", "volt-var"]
    # Read the parameter file.
    script_path = os.path.dirname(os.path.abspath(__file__))
    dss_file = pathlib.Path(script_path).joinpath("Feeders", "8500-Node", "Master.dss")

    bus = "l3104830"

    feeder_kwh_list = list()
    feeder_kvarh_list = list()
    loads_kwh_list = list()
    losses_kwh_list = list()
    pv_kwh_list = list()

    for smart_inverter_function in smart_inverter_functions_list:
        # Process for each smart inverter function
        dss.text(f"Compile [{dss_file}]")
        set_baseline()
        set_time_series_simulation()

        # Add PV system and the smart inverter function
        SmartInverterFunction(dss, bus, 12.47, 8000, 8000, smart_inverter_function)

        dss.text("solve")

        # Read Energymeter results
        energymeter_results = get_energymeter_results()
        feeder_kwh_list.append(energymeter_results[0])
        feeder_kvarh_list.append(energymeter_results[1])
        loads_kwh_list.append(energymeter_results[2])
        losses_kwh_list.append(energymeter_results[3])
        pv_kwh_list.append(energymeter_results[4])

    # dss.text("Show Eventlog") #View the event log according to the 'Show' command in the manual document.

    # Save results in a csv file
    dict_to_df = dict()
    dict_to_df["smart_inverter_function"] = smart_inverter_functions_list
    dict_to_df["feeder_kwh"] = feeder_kwh_list
    dict_to_df["feeder_kvarh"] = feeder_kvarh_list
    dict_to_df["loads_kwh"] = loads_kwh_list
    dict_to_df["losses_kwh"] = losses_kwh_list
    dict_to_df["pv_kwh"] = pv_kwh_list

    df = pd.DataFrame().from_dict(dict_to_df)

    dss.text("Show voltages LL nodes")  # Generate the result in a txt file.
    dss.text("Show voltages LN nodes")
    dss.text("Export Voltages")
    dss.text("Export Power")
    dss.text("Show power")

    # dss.text("Plot Profile  Phases=ALL")  #
    # dss.text("Plot type=circuit quantity=power")
    # dss.text("Plot Circuit Losses 1phlinestyle=3")
    # dss.text("Plot Circuit quantity=3 object=mybranchdata.csv")
    # dss.text("Plot General quantity=1 object=mybusdata.csv")

    # dss.text("Plot profile phases=Angle")
    # dss.text("Plot profile phases=LL3ph")
    # dss.text("Plot profile phases=LLall")

    # dss.text("Plot Profile Phases=Primary")
    # dss.text("plot circuit Power Max=2000 dots=n labels=n subs=n C1=$00FF0000")

    # output_file = pathlib.Path(script_path).joinpath("outputs", "results.csv")
    # df.to_csv(output_file, index=False)


if __name__ == '__main__':
    powerFlowCalculation()
