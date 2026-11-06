import os, shutil, glob, zipfile, re, sys
SRC="Parking Lot.v1i.yolo26"; DST="Parking Lot Combined"
def log(*a): print(*a, flush=True)
# resume-safe copy: skip files that already exist with the same size
for root,_,files in os.walk(SRC):
    rel=os.path.relpath(root,SRC); out=os.path.join(DST,rel); os.makedirs(out,exist_ok=True)
    for f in files:
        s=os.path.join(root,f); d=os.path.join(out,f)
        if not (os.path.exists(d) and os.path.getsize(d)==os.path.getsize(s)):
            shutil.copy2(s,d)
log("original dataset copied")
valid_clips={"8843","8844"}; cnt={"train":0,"valid":0}
for img in sorted(glob.glob("final_dataset/images/*.jpg")):
    base=os.path.basename(img)[:-4]; clip=re.match(r"my_(\d+)_",base).group(1)
    split="valid" if clip in valid_clips else "train"
    shutil.copy2(img,f"{DST}/{split}/images/{base}.jpg")
    shutil.copy2(f"final_dataset/labels/{base}.txt",f"{DST}/{split}/labels/{base}.txt"); cnt[split]+=1
for s in ("train","valid","test"):
    i=len(os.listdir(f"{DST}/{s}/images")); l=len(os.listdir(f"{DST}/{s}/labels"))
    log(s,i,"images",l,"labels","OK" if i==l else "MISMATCH")
log("added:",cnt)
with zipfile.ZipFile("Parking Lot Combined.zip","w",zipfile.ZIP_DEFLATED) as z:
    for root,_,files in os.walk(DST):
        for f in files:
            p=os.path.join(root,f); z.write(p,os.path.relpath(p,DST))
with zipfile.ZipFile("Parking Lot Combined.zip") as z:
    n=z.namelist(); log("zip MB:",round(os.path.getsize("Parking Lot Combined.zip")/1e6),"| data.yaml at top level:","data.yaml" in n,"| files:",len(n))
log("DONE")
