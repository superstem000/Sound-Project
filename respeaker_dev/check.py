import usb.core
import usb.util

# Find all devices with the given Vendor ID and Product ID
#devices = list(usb.core.find(find_all=True))
devices = list(usb.core.find(find_all=True, idVendor=0x2886, idProduct=0x0018))

if devices:
    for dev in devices:
        print(f"Device found: {dev}")
        print(f"  Device ID: {dev.idVendor}:{dev.idProduct}")
        print(f"  Device Manufacturer: {usb.util.get_string(dev, dev.iManufacturer)}")
        print(f"  Device Product: {usb.util.get_string(dev, dev.iProduct)}")
        print(f"  Device Serial Number: {usb.util.get_string(dev, dev.iSerialNumber)}")
else:
    print("No devices found")
