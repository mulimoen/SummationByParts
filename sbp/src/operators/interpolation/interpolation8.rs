use super::*;

pub struct Interpolation8;

impl Interpolation8 {
    #[rustfmt::skip]
    const F2C_DIAG: &'static [[Float; 17]] = &[
        [-2.35562114039117e-03, -1.22070312500000e-03, 1.88449691231294e-02, 1.19628906250000e-02, -6.59573919309528e-02, -5.98144531250000e-02, 1.31914783861906e-01, 2.99072265625000e-01, 3.35106520172618e-01, 2.99072265625000e-01, 1.31914783861906e-01, -5.98144531250000e-02, -6.59573919309528e-02, 1.19628906250000e-02, 1.88449691231294e-02, -1.22070312500000e-03, -2.35562114039117e-03]
    ];
    #[rustfmt::skip]
    const F2C_BLOCK: &'static [[Float; 23]] = &[
        [4.96703239864108e-01, 8.06705917431925e-01, -1.44766564766981e-02, -1.94975552500834e-01, 1.60742688137659e-01, 2.21009112212771e-01, -5.30424189394722e-01, -8.62541396704564e-01, 6.42318251728667e-03, 6.97271640112489e-01, 6.18257423430940e-01, 4.33072396957210e-01, -1.58481878611992e-01, -8.38368317429209e-01, -4.17672390622387e-01, 3.92528996502338e-01, 3.30947369649078e-01, -7.35928650870138e-02, -9.96697627809582e-02, 1.07481607311012e-02, 1.80568338088148e-02, -7.72752347871259e-04, -1.49119939947106e-03],
        [2.54878595843049e-03, 4.70070802794245e-01, 9.35891767592893e-02, 3.33862240417165e-01, -1.14183955014355e-01, -1.89982788188131e-01, 3.84844312844920e-01, 6.58280184360359e-01, 3.97045390505757e-02, -4.72524625455680e-01, -4.48926989185011e-01, -3.44029351949496e-01, 8.72844524728016e-02, 5.95394012908753e-01, 3.04844305262763e-01, -2.70963082168679e-01, -2.31087853447963e-01, 5.06082349187742e-02, 6.88018423193517e-02, -7.33227073163201e-03, -1.23334888572152e-02, 5.22750434020330e-04, 1.00876449671328e-03],
        [-2.26569732773934e-02, -9.37711842287255e-01, -5.26996809653898e-02, 1.95814305550120e+00, 1.65861481011367e+00, 2.74358371092558e+00, -3.01647084622845e+00, -5.82350161296480e+00, -1.04727678300236e+00, 3.26152098915560e+00, 3.54785958267282e+00, 3.09216402082405e+00, -3.43647933686787e-01, -4.55665472473553e+00, -2.43290788346719e+00, 1.98824139678589e+00, 1.72596012622715e+00, -3.68954584536260e-01, -5.04483632782840e-01, 5.27977630520668e-02, 8.89723433740383e-02, -3.71751659260954e-03, -7.17378410521067e-03],
        [2.16267486279437e-03, 2.76367092462810e-02, -5.52594846580351e-04, -3.35536494693914e-02, -4.92442453277949e-02, 9.34893762221034e-02, 4.57346205804429e-01, 6.65796091523885e-01, 2.47166233152219e-01, -1.58934640654239e-01, -2.58810843310230e-01, -2.68658082537024e-01, -1.80600205100413e-02, 3.08410430557262e-01, 1.77124611212295e-01, -1.25490155615361e-01, -1.12502985070320e-01, 2.29641177002937e-02, 3.17094466108080e-02, -3.21490514402454e-03, -5.43352862303885e-03, 2.21779945748522e-04, 4.27974269927444e-04],
    ];

    #[rustfmt::skip]
    const C2F_DIAG: &'static [[Float; 9]] = &[
        [-2.44140625000000e-03, 2.39257812500000e-02, -1.19628906250000e-01, 5.98144531250000e-01, 5.98144531250000e-01, -1.19628906250000e-01, 2.39257812500000e-02, -2.44140625000000e-03, 0.00000000000000e+00],
        [-4.71124228078234e-03, 3.76899382462588e-02, -1.31914783861906e-01, 2.63829567723811e-01, 6.70213040345236e-01, 2.63829567723811e-01, -1.31914783861906e-01, 3.76899382462588e-02, -4.71124228078234e-03],
    ];
    #[rustfmt::skip]
    const C2F_BLOCK: &'static [[Float; 16]] = &[
        [9.93406479728215e-01, 2.63740810871388e-02, -3.95611216307082e-02, 2.63740810871388e-02, -6.59352027178470e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [3.11839598602877e-01, 9.40141605588491e-01, -3.16462408382736e-01, 6.51416055884908e-02, -6.60401397122704e-04, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [-3.31635914674324e-02, 1.10925881915128e+00, -1.05399361930780e-01, -7.71891444099258e-03, 6.04185954063824e-02, -2.33955467184537e-02, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [-6.39519875380412e-02, 5.66572075303674e-01, 5.60731574165718e-01, -6.71072989387827e-02, 5.49151185592384e-03, -1.73587484849126e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [2.29709689957704e-01, -8.44242373309120e-01, 2.06933277237064e+00, -4.29101155545423e-01, -1.31914783861906e-01, 1.16755671676913e-01, -1.05398212888044e-02, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [1.01954337766153e-01, -4.53444105476147e-01, 1.10496991032767e+00, 2.62974658127715e-01, -3.86174480800107e-02, 2.39257812500000e-02, -1.76313391538362e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [-3.38823560499617e-01, 1.27188934494977e+00, -1.68223289417986e+00, 1.78135907280824e+00, 1.17930369056755e-01, -1.82662005975752e-01, 3.76899382462588e-02, -5.15026440579746e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [-5.04006504823111e-01, 1.99012773182278e+00, -2.97082521952307e+00, 2.37221225007671e+00, 2.44576227277174e-01, -1.51529363117418e-01, 2.18862845369385e-02, -2.44140625000000e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [3.78827327147319e-03, 1.21156068183631e-01, -5.39248841565280e-01, 8.88865980757861e-01, 2.76602322166401e-01, 3.37302045431818e-01, -1.21796336850761e-01, 3.80417308856396e-02, -4.71124228078234e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [4.11237810864861e-01, -1.44188113271454e+00, 1.67937592400445e+00, -5.71565110006450e-01, 2.46859067752038e-01, 7.64718585544162e-01, -1.10452840357651e-01, 2.41491011631342e-02, -2.44140625000000e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [3.64636699295086e-01, -1.36987433204775e+00, 1.82681331338964e+00, -9.30742646905333e-01, 1.08884588474992e-01, 8.56857066226101e-01, 2.43592673701522e-01, -1.33146058099739e-01, 3.76899382462588e-02, -4.71124228078234e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [2.55418023945373e-01, -1.04978535499102e+00, 1.59217304653592e+00, -9.66155558456609e-01, -4.93718135504075e-02, 7.64718585544162e-01, 5.52264201788255e-01, -1.20745505815671e-01, 2.39257812500000e-02, -2.44140625000000e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [-9.34696566916614e-02, 2.66343378566745e-01, -1.76946298394627e-01, -6.49479406569206e-02, -5.44422942374959e-02, 3.37302045431818e-01, 6.18804737679094e-01, 2.66292116199477e-01, -1.31914783861906e-01, 3.76899382462588e-02, -4.71124228078234e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [-4.94454000025620e-01, 1.81680984968021e+00, -2.34624773661296e+00, 1.10911404174051e+00, 9.87436271008150e-03, -1.52943717108832e-01, 5.52264201788255e-01, 6.03727529078354e-01, -1.19628906250000e-01, 2.39257812500000e-02, -2.44140625000000e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [-2.46335387382934e-01, 9.30214487234333e-01, -1.25271826807198e+00, 6.36980380587063e-01, 1.55549412107131e-02, -1.68651022715909e-01, 2.43592673701522e-01, 6.76468715609812e-01, 2.63829567723811e-01, -1.31914783861906e-01, 3.76899382462588e-02, -4.71124228078234e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [2.31506282395997e-01, -8.26827925559284e-01, 1.02375693548293e+00, -4.51291136430475e-01, -1.00758803164097e-03, 3.05887434217665e-02, -1.10452840357651e-01, 6.03727529078354e-01, 5.98144531250000e-01, -1.19628906250000e-01, 2.39257812500000e-02, -2.44140625000000e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [1.95186587230214e-01, -7.05151007875617e-01, 8.88706800114134e-01, -4.04586317828987e-01, -1.94436765133914e-03, 4.81860064902597e-02, -1.21796336850761e-01, 2.66292116199477e-01, 6.70213040345236e-01, 2.63829567723811e-01, -1.31914783861906e-01, 3.76899382462588e-02, -4.71124228078234e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [-4.34036994947538e-02, 1.54428055509268e-01, -1.89976838530687e-01, 8.25841893594732e-02, 0.00000000000000e+00, -3.12130034915985e-03, 2.20905680715302e-02, -1.20745505815671e-01, 5.98144531250000e-01, 5.98144531250000e-01, -1.19628906250000e-01, 2.39257812500000e-02, -2.44140625000000e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [-5.87833674819310e-02, 2.09944779577586e-01, -2.59761525302692e-01, 1.14034380835697e-01, 0.00000000000000e+00, -6.02325081128246e-03, 3.47989533859317e-02, -1.33146058099739e-01, 2.63829567723811e-01, 6.70213040345236e-01, 2.63829567723811e-01, -1.31914783861906e-01, 3.76899382462588e-02, -4.71124228078234e-03, 0.00000000000000e+00, 0.00000000000000e+00],
        [6.33906477132592e-03, -2.23739933505050e-02, 2.71858719921617e-02, -1.15615299769810e-02, 0.00000000000000e+00, 0.00000000000000e+00, -2.25413959913573e-03, 2.41491011631342e-02, -1.19628906250000e-01, 5.98144531250000e-01, 5.98144531250000e-01, -1.19628906250000e-01, 2.39257812500000e-02, -2.44140625000000e-03, 0.00000000000000e+00, 0.00000000000000e+00],
        [1.06495838630259e-02, -3.76349166281317e-02, 4.58123715473316e-02, -1.95402045291476e-02, 0.00000000000000e+00, 0.00000000000000e+00, -4.34986917324146e-03, 3.80417308856396e-02, -1.31914783861906e-01, 2.63829567723811e-01, 6.70213040345236e-01, 2.63829567723811e-01, -1.31914783861906e-01, 3.76899382462588e-02, -4.71124228078234e-03, 0.00000000000000e+00],
        [-4.55754924763598e-04, 1.59514223667259e-03, -1.91417068400711e-03, 7.97571118336296e-04, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, -2.46419399623818e-03, 2.39257812500000e-02, -1.19628906250000e-01, 5.98144531250000e-01, 5.98144531250000e-01, -1.19628906250000e-01, 2.39257812500000e-02, -2.44140625000000e-03, 0.00000000000000e+00],
        [-8.79481598452137e-04, 3.07818559458248e-03, -3.69382271349897e-03, 1.53909279729124e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, -4.75521636070495e-03, 3.76899382462588e-02, -1.31914783861906e-01, 2.63829567723811e-01, 6.70213040345236e-01, 2.63829567723811e-01, -1.31914783861906e-01, 3.76899382462588e-02, -4.71124228078234e-03],
    ];
}

impl InterpolationOperator for Interpolation8 {
    fn fine2coarse(&self, fine: ArrayView1<Float>, coarse: ArrayViewMut1<Float>) {
        assert_eq!(fine.len(), 2 * coarse.len() - 1);
        use ndarray::prelude::*;
        let block = Array::from_iter(Self::F2C_BLOCK.iter().flatten().copied())
            .into_shape((Self::F2C_BLOCK.len(), Self::F2C_BLOCK[0].len()))
            .unwrap();
        let diag = Array::from_iter(Self::F2C_DIAG.iter().flatten().copied())
            .into_shape((Self::F2C_DIAG.len(), Self::F2C_DIAG[0].len()))
            .unwrap();
        super::interpolate(fine, coarse, block.view(), diag.view(), (0, 2))
    }
    fn coarse2fine(&self, coarse: ArrayView1<Float>, fine: ArrayViewMut1<Float>) {
        assert_eq!(fine.len(), 2 * coarse.len() - 1);
        use ndarray::prelude::*;
        let block = Array::from_iter(Self::C2F_BLOCK.iter().flatten().copied())
            .into_shape((Self::C2F_BLOCK.len(), Self::C2F_BLOCK[0].len()))
            .unwrap();
        let diag = Array::from_iter(Self::C2F_DIAG.iter().flatten().copied())
            .into_shape((Self::C2F_DIAG.len(), Self::C2F_DIAG[0].len()))
            .unwrap();
        super::interpolate(coarse, fine, block.view(), diag.view(), (8, 1))
    }
}

#[test]
fn test_inter8() {
    test_interpolation_operator(Interpolation8, false);
}
