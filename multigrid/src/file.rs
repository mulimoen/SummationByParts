use super::*;

pub struct OutputThread {
    rx: Option<std::sync::mpsc::Receiver<Vec<euler::Field>>>,
    tx: Option<std::sync::mpsc::SyncSender<(u64, Vec<euler::Field>)>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl OutputThread {
    pub fn new(file: File) -> Self {
        // Pingpong back and forth a number of Vec<Field> to be used for the
        // output. The sync_channel applies some backpressure
        let (tx_thread, rx) = std::sync::mpsc::channel::<Vec<euler::Field>>();
        let (tx, rx_thread) = std::sync::mpsc::sync_channel::<(u64, Vec<euler::Field>)>(3);
        let thread = std::thread::Builder::new()
            .name("multigrid_output".to_owned())
            .spawn(move || {
                let mut times = Vec::<u64>::new();

                for (ntime, fields) in rx_thread.iter() {
                    if !times.contains(&ntime) {
                        file.add_timestep(ntime, fields.as_slice()).unwrap();
                        times.push(ntime);
                    }
                    tx_thread.send(fields).unwrap();
                }
            })
            .unwrap();

        Self {
            tx: Some(tx),
            rx: Some(rx),
            thread: Some(thread),
        }
    }

    pub fn add_timestep(&mut self, ntime: u64, fields: &[euler::Field]) {
        match self.rx.as_ref().unwrap().try_recv() {
            Ok(mut copy_fields) => {
                for (from, to) in fields.iter().zip(copy_fields.iter_mut()) {
                    use integrate::Integrable;
                    euler::Field::assign(to, from);
                }
                self.tx
                    .as_ref()
                    .unwrap()
                    .send((ntime, copy_fields))
                    .unwrap();
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                let fields = fields.to_vec();
                self.tx.as_ref().unwrap().send((ntime, fields)).unwrap();
            }
            Err(e) => panic!("{:?}", e),
        };
    }
}

impl Drop for OutputThread {
    fn drop(&mut self) {
        let tx = self.tx.take();
        std::mem::drop(tx);
        let thread = self.thread.take().unwrap();
        thread.join().unwrap();
    }
}

#[derive(Debug, Clone)]
pub struct File(hdf5::File, Vec<String>);

impl File {
    pub fn create<P: AsRef<std::path::Path>>(
        path: P,
        grids: &[sbp::grid::Grid],
        names: Vec<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        assert_eq!(grids.len(), names.len());
        let file = hdf5::File::create(path.as_ref())?;
        let _tds = file
            .new_dataset::<u64>()
            .resizable(true)
            .chunk((1,))
            .create("t", (0,))?;

        for (name, grid) in names.iter().zip(grids.iter()) {
            let g = file.create_group(name)?;
            g.link_soft("/t", "t").unwrap();

            let add_dim = |name| {
                g.new_dataset::<Float>()
                    .chunk((grid.ny(), grid.nx()))
                    .gzip(9)
                    .create(name, (grid.ny(), grid.nx()))
            };
            let xds = add_dim("x")?;
            xds.write(grid.x())?;
            let yds = add_dim("y")?;
            yds.write(grid.y())?;

            let add_var = |name| {
                g.new_dataset::<Float>()
                    .gzip(3)
                    .shuffle(true)
                    .chunk((1, grid.ny(), grid.nx()))
                    .resizable(true)
                    .create(name, (0, grid.ny(), grid.nx()))
            };
            add_var("rho")?;
            add_var("rhou")?;
            add_var("rhov")?;
            add_var("e")?;
        }

        Ok(Self(file, names))
    }

    pub fn add_timestep(
        &self,
        t: u64,
        fields: &[euler::Field],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = &self.0;
        let tds = file.dataset("t")?;
        let tpos = tds.size();
        tds.resize((tpos + 1,))?;
        tds.write_slice(&[t], ndarray::s![tpos..tpos + 1])?;

        for (groupname, fnow) in self.1.iter().zip(fields.iter()) {
            let g = file.group(groupname)?;
            let (tpos, ny, nx) = {
                let ds = g.dataset("rho")?;
                let shape = ds.shape();
                (shape[0], shape[1], shape[2])
            };

            let rhods = g.dataset("rho")?;
            let rhouds = g.dataset("rhou")?;
            let rhovds = g.dataset("rhov")?;
            let eds = g.dataset("e")?;

            let (rho, rhou, rhov, e) = fnow.components();
            rhods.resize((tpos + 1, ny, nx))?;
            rhods.write_slice(rho, ndarray::s![tpos, .., ..])?;

            rhouds.resize((tpos + 1, ny, nx))?;
            rhouds.write_slice(rhou, ndarray::s![tpos, .., ..])?;

            rhovds.resize((tpos + 1, ny, nx))?;
            rhovds.write_slice(rhov, ndarray::s![tpos, .., ..])?;

            eds.resize((tpos + 1, ny, nx))?;
            eds.write_slice(e, ndarray::s![tpos, .., ..])?;
        }
        Ok(())
    }
}
