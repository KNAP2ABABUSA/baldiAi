use std::{collections::HashMap, fs::{self, OpenOptions}, io::Write};
use serde::Deserialize;
use burn::{Tensor, backend::{Autodiff, Wgpu, wgpu::WgpuDevice}, data::{dataloader::{DataLoaderBuilder, batcher::Batcher}, dataset::InMemDataset}, module::Module, nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, Lstm, LstmConfig, loss::{CrossEntropyLoss, CrossEntropyLossConfig}}, optim::{AdamWConfig, GradientsParams, Optimizer}, tensor::{Float, Shape, TensorData, backend::Backend, cast::ToElement}};
use once_cell::sync::Lazy;
use rand::{distr::weighted::WeightedIndex, prelude::Distribution, rng};
use chrono::Local;

#[derive(Debug, Deserialize, Clone)]
pub struct Expl {
    primer: Vec<usize>,
    chel: usize,
}
static SLOVHS: Lazy<HashMap<String, usize>> = Lazy::new(||{let contentek = fs::read_to_string("slovar.json").unwrap_or_else(|e| panic!("Не прочитано slovar.json: {}", e));
    serde_json::from_str(&contentek).unwrap_or_else(|e| panic!("С парсингом что то не так: {}", e))});
static CARTA: Lazy<WgpuDevice> = Lazy::new(|| {burn::backend::wgpu::WgpuDevice::default()});
const SQLL: usize = 15;
pub struct Btch;
impl Btch{
    pub fn nw() -> Self{
        Self}}

impl Batcher<Wgpu, Expl, (Tensor<Autodiff<Wgpu>, 2, Float>, Tensor<Autodiff<Wgpu>, 1, Float>)> for Btch{
    fn batch(&self, items: Vec<Expl>, device: &WgpuDevice) -> (Tensor<Autodiff<Wgpu>, 2, Float>, Tensor<Autodiff<Wgpu>, 1, Float>){
        let mxlen = items.iter().map(|item| item.primer.len()).max().unwrap_or(0);
        let mut inf = Vec::with_capacity(items.len() * mxlen);
        let mut cheli = Vec::with_capacity(items.len());
        for i in &items{
            let mut bz = i.primer.clone();
            bz.resize(mxlen, 0);
            inf.extend(bz);
            cheli.push(i.chel as i64);}
        let inf_f32: Vec<f32> = inf.iter().map(|&x| x as f32).collect();
        let inputs = Tensor::<Autodiff<Wgpu>, 2, Float>::from_data(TensorData::new(inf_f32.clone(), Shape::new([items.len(), mxlen])),device,);
        let cheli_f32: Vec<f32> = cheli.iter().map(|&x| x as f32).collect();
        let targets = Tensor::<Autodiff<Wgpu>, 1, Float>::from_data(TensorData::new(cheli_f32, Shape::new([items.len()])), device);
        println!("items.len(): {}, mxlen: {}", items.len(), mxlen);
        println!("inf_f32.len(): {}", inf_f32.len());
        (inputs, targets)}}

#[derive(Module, Debug)]
pub struct Madelka<B: Backend> {
    emb: Embedding<B>,
    lstm: Lstm<B>,
    drop: Dropout,
    vixod: Linear<B>}

impl<B: Backend> Madelka<B>{
    fn newmodel(device: &B::Device) -> Self{
        let emb = EmbeddingConfig::new(SLOVHS.len(), 32).init(device);
        let lstm = LstmConfig::new(32, 32, true).with_batch_first(true).init(device);
        let drop = DropoutConfig::new(0.15).init();
        let vixod = LinearConfig::new(32, SLOVHS.len()).init(device);
        Self {emb, lstm, drop, vixod}}
    pub fn goahead(&self, input: Tensor<B, 2, Float>) -> Tensor<B, 2>{
        let embgot = self.emb.forward(input.int());
        let (outp, _stati) = self.lstm.forward(embgot, None);
        let lstout = outp.clone().slice([0.., outp.dims()[1] - 1.., 0..]).squeeze_dim::<2>(1);
        self.vixod.forward(self.drop.forward(lstout))}}

fn degenerat(madel: Madelka<Autodiff<Wgpu>>, zatrav: Vec<String>, leng: i32, qcold: f32) -> Vec<usize>{
    let mut veci: Vec<usize> = Vec::new();
    let mut uw: Vec<usize> = Vec::new();
    for slov in zatrav{
    match SLOVHS.get(&slov){
    Some(&znach) => {veci.push(znach); uw.push(znach)}
    None => veci.push(1.to_usize())}}
    for _ in 0..leng{
    let mut razr: Vec<usize> =
if veci.len() < SQLL{
    veci.clone()}
else{
    veci[veci.len() - SQLL..].to_vec()};
while razr.len() < SQLL {
    razr.insert(0, 0);}
    let razrf: Vec<f32> = razr.iter().map(|&x| x as f32).collect();
    let tnssr = Tensor::<Autodiff<Wgpu>, 2, Float>::from_data(TensorData::new(razrf, Shape::new([1, SQLL])), &*CARTA);
    let predskaz = madel.goahead(tnssr);
    let probs = burn::tensor::activation::softmax(predskaz / qcold, 1);
    let mut probsasv: Vec<f32> = probs.to_data().into_vec().expect("Не получилось конвертировать в Vecf32");
    for w in &uw{
        probsasv[*w] *= 0.5;}
    let sum: f32 = probsasv.iter().sum();
    for p in &mut probsasv{*p /= sum;}
    let distr = WeightedIndex::new(&probsasv).expect("Опять же, не получилось сделать взвешенный индекс");
    let nextwrd = distr.sample(&mut rng());
    veci.push(nextwrd);}
    veci
}
fn startmodel() -> Result<(), Box<dyn std::error::Error>>{
    type Bck = Autodiff<Wgpu>;
    let logepoh = [0, 1, 2, 3, 4, 5, 10, 100, 500, 1000, 5000, 10000];//КАКИЕ ЭПОХИ БУДУТ ЛОГИРОВАТЬСЯ
    let her: Vec<Expl> = serde_json::from_str(&fs::read_to_string("chel.json")?)?;
    let _dtset = InMemDataset::new(her);
    let spid = rand::random::<u64>();
    let drload = DataLoaderBuilder::new(Btch::nw()).batch_size(8).shuffle(spid).num_workers(2).build(_dtset);
    let losi: CrossEntropyLoss<Bck> = CrossEntropyLossConfig::new().init(&*CARTA);
    let mut trueadam = AdamWConfig::new().with_weight_decay(5e-3).init::<Bck, Madelka<Bck>>();
    let mut samamodel = Madelka::<Bck>::newmodel(&CARTA);
    for epoha in 0..10001{//ВОТ ТУТ МЕНЯЙ КОЛ ВО ЭПОХ
        for (inputs, targets) in drload.iter(){
            let predskaz = samamodel.goahead(inputs);
            let oshibka = losi.forward(predskaz, targets.int());
            let gards = oshibka.backward();
            samamodel = trueadam.step(0.09, samamodel.clone(), GradientsParams::from_grads(gards, &samamodel));
            println!("Потери: {} на эпоху {}", oshibka.into_scalar(), epoha);}
        if logepoh.contains(&epoha){
            let otvet = degenerat(samamodel.clone(), vec!["три".to_string()], 60, 1.5);
            let slinv: HashMap<usize, String> = SLOVHS.iter().map(|(word, &idx)| (idx, word.clone())).collect();
            let wrs: Vec<String> = otvet.iter().map(|&idx| slinv.get(&idx).unwrap_or(&"<UNK>".to_string()).clone()).collect();
            let txt = wrs.join(" ");
            let mut logfile = OpenOptions::new().create(true).append(true).open("log.md")?;
            writeln!(logfile, "# ЭПОХА №{}   ВРЕМЯ: {:?}", epoha, Local::now().format("%H:%M"))?;
            writeln!(logfile, "ГЕНЕРИРУЕТСЯ:\n > {}", txt)?;}}
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>>{
    println!("Привет! Тут будет скрипт для создания словаря и целей");
    startmodel()
}