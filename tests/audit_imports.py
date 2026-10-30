import pefile
import os
import sys

dll_path = os.path.abspath("astrosnail/rust/astrosnail_rs/target/release/astrosnail_rs.dll")
if not os.path.exists(dll_path):
    print(f"DLL not found: {dll_path}")
    sys.exit(1)

pe = pefile.PE(dll_path)
print("=== DLL Import Directory Audit ===")
imports = {}
for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
    dll_name = entry.dll.decode('utf-8', errors='ignore')
    funcs = []
    for imp in entry.imports:
        name = imp.name.decode('utf-8', errors='ignore') if imp.name else f"Ordinal({imp.ordinal})"
        funcs.append(name)
    imports[dll_name] = funcs
    print(f"Imported DLL: {dll_name} ({len(funcs)} functions)")
    for f in funcs:
        print(f"   - {f}")

# Check for heap allocation functions
alloc_funcs = ["malloc", "calloc", "realloc", "free", "_malloc_base", "_calloc_base", "_realloc_base", "_free_base", "HeapAlloc", "HeapFree", "HeapReAlloc"]
found_allocs = []
for dll_name, funcs in imports.items():
    for f in funcs:
        for target in alloc_funcs:
            if target.lower() in f.lower():
                found_allocs.append((dll_name, f))

print("\n=== Heap Allocation Audit Results ===")
if found_allocs:
    print(f"WARNING / DETECTED heap functions: {found_allocs}")
else:
    print("ZERO heap allocation functions imported in astrosnail_rs.dll!")
