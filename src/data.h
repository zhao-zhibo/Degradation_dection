#include <vector>

#include <Eigen/Dense>

namespace data {

const std::vector<Eigen::Vector3d> points{
    Eigen::Vector3d(11.852234575434, -14.958526478893, 2.211156876723),
    Eigen::Vector3d(8.318883123980, -15.141624601719, 4.023703308503),
    Eigen::Vector3d(2.871273644142, -15.040119016036, -3.096382227940),
    Eigen::Vector3d(4.339235069551, -15.187819310360, -1.027000309302),
    Eigen::Vector3d(9.549127772867, -15.031169323983, 0.492730778279),
    Eigen::Vector3d(-8.616208839753, -14.883968826667, -6.909127373182),
    Eigen::Vector3d(-12.106156593892, -15.051667773378, -3.671317824859),
    Eigen::Vector3d(-12.312542033994, -14.870323162102, 4.193830862441),
    Eigen::Vector3d(8.455469141218, -15.027562977430, 4.186314136229),
    Eigen::Vector3d(10.173931009411, -14.983906502970, 6.552460919272),
    Eigen::Vector3d(4.836771780905, -14.897073201023, 5.239500052480),
    Eigen::Vector3d(-11.193738366988, -14.820656627673, -5.802719671064),
    Eigen::Vector3d(-6.282541442690, -15.006517074612, 6.135982722741),
    Eigen::Vector3d(-3.193737880022, -15.040375466987, 4.426863612749),
    Eigen::Vector3d(7.087850746709, -15.068306764481, -3.662802022785),
    Eigen::Vector3d(5.462577707388, -15.040192292792, -0.987708377113),
    Eigen::Vector3d(9.144104352691, -15.213903196414, -2.972196342268),
    Eigen::Vector3d(9.487361017775, -14.966978650047, -7.137574184000),
    Eigen::Vector3d(5.115725087410, -15.027102757914, -4.313617117937),
    Eigen::Vector3d(-1.047445567640, -15.038096386553, 0.728130647635),
    Eigen::Vector3d(3.272186676050, 15.012962792689, 1.814840074354),
    Eigen::Vector3d(5.409765151454, 15.001174906023, -4.193137679094),
    Eigen::Vector3d(-8.089371000147, 14.823891126757, -4.726880314205),
    Eigen::Vector3d(-7.013105540290, 14.988493830593, 3.874207395844),
    Eigen::Vector3d(-0.197873651403, 14.986484906820, 3.122576040134),
    Eigen::Vector3d(5.709597344726, 14.957890483626, -4.716501292225),
    Eigen::Vector3d(-3.255902962332, 14.919747473036, -0.014202956302),
    Eigen::Vector3d(-11.745595708327, 15.058356503759, 1.423986530161),
    Eigen::Vector3d(-12.360575254996, 15.043594180187, -2.059589336149),
    Eigen::Vector3d(-2.678610127053, 15.117200247340, 1.528019458686),
    Eigen::Vector3d(5.976716261678, 14.905749518100, 3.075704767757),
    Eigen::Vector3d(0.300503238520, 14.955345851927, 5.976930544330),
    Eigen::Vector3d(-1.653880091442, 15.012440849702, 4.086049680720),
    Eigen::Vector3d(5.713520602205, 14.954555561103, -7.227706205788),
    Eigen::Vector3d(9.052766543512, 14.982520338797, 0.871398840520),
    Eigen::Vector3d(-1.553028405328, 15.076571114808, -0.756156587275),
    Eigen::Vector3d(-10.930142941801, 15.054105013457, 6.443216079332),
    Eigen::Vector3d(-11.051102454678, 14.858192662854, 4.357557152114),
    Eigen::Vector3d(7.947762112078, 15.040202696013, 1.420085928197),
    Eigen::Vector3d(9.731902524751, 14.963189154592, -6.937996418779),
    Eigen::Vector3d(-12.605227282069, 1.629936974591, 7.056947804874),
    Eigen::Vector3d(-12.644891825063, 2.602589866302, -6.202190923954),
    Eigen::Vector3d(-12.596018369687, 12.041817395406, -6.199022970663),
    Eigen::Vector3d(-12.505425823160, 0.438430740676, 4.708303629892),
    Eigen::Vector3d(-12.616051470422, -12.304947199726, 0.806106945212),
    Eigen::Vector3d(-12.388288737725, -7.556984126394, 4.362928877260),
    Eigen::Vector3d(-12.577190921498, -12.911585098424, -2.990811550683),
    Eigen::Vector3d(-12.499915343412, -14.589862061109, -0.739896449998),
    Eigen::Vector3d(-12.527580162598, 7.125645099776, 4.684466628168),
    Eigen::Vector3d(-12.436292991907, 8.729743159038, 1.271789069445),
    Eigen::Vector3d(-12.693168597375, -5.425982311086, 3.496617976126),
    Eigen::Vector3d(-12.573460574949, 12.215051995474, -6.048782162308),
    Eigen::Vector3d(-12.577307540623, -12.006428528079, -2.033183234518),
    Eigen::Vector3d(-12.438823339249, -1.410646879891, -7.047089895827),
    Eigen::Vector3d(-12.498373405161, -4.992032743443, 1.595235422977),
    Eigen::Vector3d(-12.262752411313, 8.050315124855, 6.199307863835),
    Eigen::Vector3d(-12.487442705026, 6.194458427016, 1.551006955064),
    Eigen::Vector3d(-12.702086752586, 9.320026822071, 4.200704744528),
    Eigen::Vector3d(-12.620128072538, 0.695618670332, -6.071128280347),
    Eigen::Vector3d(-12.497513372072, -1.734170742250, -4.906015742186),
    Eigen::Vector3d(12.563354435498, -14.241148307266, -4.121851960161),
    Eigen::Vector3d(12.547049197634, -0.760236256184, -1.751073229791),
    Eigen::Vector3d(12.440606028017, 9.219417390437, -0.164714250697),
    Eigen::Vector3d(12.465192673661, 9.103525302952, -1.653930482784),
    Eigen::Vector3d(12.512091546618, 5.869995974446, 5.853671274903),
    Eigen::Vector3d(12.481451925780, -2.552706413723, -3.709156690493),
    Eigen::Vector3d(12.525510337245, -8.987431237964, -1.569342710168),
    Eigen::Vector3d(12.538861337378, -7.425913676763, -2.560550797047),
    Eigen::Vector3d(12.626096722198, -5.638857425328, -2.768466812572),
    Eigen::Vector3d(12.555778236743, 12.216636340113, -1.271164875133),
    Eigen::Vector3d(12.642321005933, 12.458003703549, -6.915588257596),
    Eigen::Vector3d(12.543975217926, -11.572098638835, 3.963313982696),
    Eigen::Vector3d(12.562994960876, -6.370438082838, 6.863263396383),
    Eigen::Vector3d(12.467113213414, 2.593421455768, 1.631049341319),
    Eigen::Vector3d(12.420632674130, 5.795596179955, -1.606238402568),
    Eigen::Vector3d(12.558638378847, 14.950109049688, 1.335605206903),
    Eigen::Vector3d(12.636343267864, -12.622240326276, 1.549226081733),
    Eigen::Vector3d(12.490244938862, 5.802516956799, 6.144211324440),
    Eigen::Vector3d(12.615697265001, -2.700843742266, 4.725817053610),
    Eigen::Vector3d(12.524548309526, -14.542735364561, -1.980853355593),
    Eigen::Vector3d(9.262303632755, 12.072588111782, -7.483584292120),
    Eigen::Vector3d(0.568725054876, -10.572040406915, -7.488139584853),
    Eigen::Vector3d(7.667155176225, 12.511250966428, -7.394990810486),
    Eigen::Vector3d(-5.191646111158, -8.472507110131, -7.612818883626),
    Eigen::Vector3d(12.307033345284, 11.625439711533, -7.328074108367),
    Eigen::Vector3d(-2.974747224518, 11.285873175413, -7.654300637088),
    Eigen::Vector3d(-11.794308027998, -5.723214038626, -7.434823849695),
    Eigen::Vector3d(10.873963300371, -5.476391628094, -7.522322831737),
    Eigen::Vector3d(1.777263319153, -11.358413711701, -7.453435912347),
    Eigen::Vector3d(-4.221471433348, 7.008853647830, -7.522492665504),
    Eigen::Vector3d(11.407599052322, 0.220304807562, -7.488826821074),
    Eigen::Vector3d(11.769695119082, 5.666581534162, -7.637650772920),
    Eigen::Vector3d(10.720514987544, 13.573393666296, -7.453328048629),
    Eigen::Vector3d(-2.575957974036, 14.195824131168, -7.482321120296),
    Eigen::Vector3d(-4.335644957475, 11.753826459702, -7.445900160163),
    Eigen::Vector3d(-9.996692281482, -1.279677263404, -7.594240971184),
    Eigen::Vector3d(-6.844198301009, 9.859950754390, -7.440526505940),
    Eigen::Vector3d(10.471696650404, -13.071536148770, -7.380688920800),
    Eigen::Vector3d(-4.628313701766, 14.846209630027, -7.499017877562),
    Eigen::Vector3d(-10.539845267493, -14.543361267724, -7.516612549665),
    Eigen::Vector3d(-8.131423102261, 0.757789590435, 7.564200156533),
    Eigen::Vector3d(-10.255438092964, 1.941648838182, 7.549294302174),
    Eigen::Vector3d(0.868478810813, 11.810166698835, 7.486090495192),
    Eigen::Vector3d(-0.020624105649, -5.448421768511, 7.628182622732),
    Eigen::Vector3d(10.947868816713, 12.101822044344, 7.400808312173),
    Eigen::Vector3d(3.807645436606, -14.441516569535, 7.585822216937),
    Eigen::Vector3d(12.116906708569, -4.224941497491, 7.400462285435),
    Eigen::Vector3d(-8.699905875593, -13.431048156095, 7.613985832983),
    Eigen::Vector3d(-8.645687198143, -13.973412997774, 7.551417042150),
    Eigen::Vector3d(-8.567590143552, -6.696013074361, 7.554540544078),
    Eigen::Vector3d(4.665113469546, 4.794666201863, 7.405970299177),
    Eigen::Vector3d(-1.695922738157, 6.066163284633, 7.523911392303),
    Eigen::Vector3d(5.921134791924, 13.203190634334, 7.615204795908),
    Eigen::Vector3d(6.774990013513, -14.639565834398, 7.672187562289),
    Eigen::Vector3d(7.296573336440, 6.141348566504, 7.521792301000),
    Eigen::Vector3d(10.727220003904, 11.512855815731, 7.691256543379),
    Eigen::Vector3d(-0.452291144794, -9.896970917145, 7.695822414707),
    Eigen::Vector3d(1.496484345670, 0.117860256513, 7.396017330751),
    Eigen::Vector3d(-3.467096612295, 13.467499215285, 7.517417144135),
    Eigen::Vector3d(-4.175483062272, -9.725850630337, 7.411076674572)};

const std::vector<Eigen::Vector3d> normals{
    Eigen::Vector3d(0.003371572701, 0.999992847614, 0.001713831529),
    Eigen::Vector3d(0.037975117806, 0.999115402464, 0.018063858581),
    Eigen::Vector3d(0.055973785494, 0.996468939856, 0.062582643276),
    Eigen::Vector3d(-0.017539285438, 0.995140290018, -0.096892603692),
    Eigen::Vector3d(-0.000604156159, 0.999024773085, 0.044149040291),
    Eigen::Vector3d(-0.012945985039, 0.999852952010, 0.011246147271),
    Eigen::Vector3d(0.048814884753, 0.998312386073, -0.031456109737),
    Eigen::Vector3d(-0.113388630538, 0.993531726434, 0.006142233567),
    Eigen::Vector3d(0.011753649331, 0.999920991496, -0.004456735777),
    Eigen::Vector3d(0.033115905468, 0.999399687869, 0.010178442525),
    Eigen::Vector3d(0.015899294957, 0.999269765816, -0.034744028935),
    Eigen::Vector3d(0.028600594374, 0.999422103456, 0.018370223893),
    Eigen::Vector3d(0.051422599615, 0.997221443609, 0.053899059866),
    Eigen::Vector3d(0.031205505591, 0.999510808100, 0.002088279811),
    Eigen::Vector3d(-0.066089736462, 0.995823896379, 0.062983443344),
    Eigen::Vector3d(0.037816172277, 0.999138051189, 0.017119923490),
    Eigen::Vector3d(0.087964831204, 0.995511002172, -0.034928971157),
    Eigen::Vector3d(-0.058551517772, 0.998153811813, 0.016145827023),
    Eigen::Vector3d(0.026550000678, 0.999503487296, -0.016966919189),
    Eigen::Vector3d(-0.060733552487, 0.997623272742, -0.032545987247),
    Eigen::Vector3d(0.062451006213, 0.997317685999, -0.038174664600),
    Eigen::Vector3d(-0.033102773803, 0.999284415642, -0.018299262896),
    Eigen::Vector3d(-0.003673217902, 0.998913794656, 0.046451461939),
    Eigen::Vector3d(-0.096376791672, 0.994563002330, -0.039445512085),
    Eigen::Vector3d(-0.034953425047, 0.997948640174, 0.053635526029),
    Eigen::Vector3d(0.047694253314, 0.998327613054, -0.032668566171),
    Eigen::Vector3d(0.020105020041, 0.998673100375, 0.047411251372),
    Eigen::Vector3d(-0.043278425152, 0.997138572798, 0.061981001562),
    Eigen::Vector3d(0.064828107572, 0.997666783918, -0.021408053045),
    Eigen::Vector3d(0.057426891033, 0.996976311298, 0.052348704820),
    Eigen::Vector3d(-0.034879378919, 0.998384154064, -0.044861005786),
    Eigen::Vector3d(-0.078654155089, 0.992495033938, -0.093632961585),
    Eigen::Vector3d(0.029796756657, 0.998215975846, -0.051739915524),
    Eigen::Vector3d(-0.046075190789, 0.996534139647, -0.069258828396),
    Eigen::Vector3d(-0.048318490312, 0.995659698745, 0.079542993336),
    Eigen::Vector3d(0.021145868395, 0.998540646123, -0.049693362641),
    Eigen::Vector3d(-0.013425717476, 0.999460812941, 0.029963870000),
    Eigen::Vector3d(0.040623766591, 0.999093798755, -0.012700034564),
    Eigen::Vector3d(0.038679209982, 0.999105830747, 0.017072131727),
    Eigen::Vector3d(0.048337028770, 0.998667596432, -0.018071067671),
    Eigen::Vector3d(0.998166449192, 0.047086056789, 0.038034759951),
    Eigen::Vector3d(0.996196396063, -0.022995351029, 0.084047333700),
    Eigen::Vector3d(0.998487603470, -0.052520935537, 0.016249832219),
    Eigen::Vector3d(0.999024787128, 0.020982097700, 0.038848761630),
    Eigen::Vector3d(0.996767606908, -0.043962366771, -0.067243201340),
    Eigen::Vector3d(0.995358131346, 0.096239559587, 0.000370856186),
    Eigen::Vector3d(0.999683752819, -0.025124819471, -0.001066675787),
    Eigen::Vector3d(0.999226323367, 0.004523968992, -0.039067741100),
    Eigen::Vector3d(0.996989322752, -0.076323127360, 0.013677373628),
    Eigen::Vector3d(0.998321592271, 0.051365791468, 0.026749838734),
    Eigen::Vector3d(0.998001115472, -0.060623535210, -0.017848263086),
    Eigen::Vector3d(0.999316306012, -0.002963856902, 0.036852897997),
    Eigen::Vector3d(0.999383217527, 0.032729806230, 0.012725734151),
    Eigen::Vector3d(0.999101213842, -0.001010862903, 0.042376203890),
    Eigen::Vector3d(0.999370267596, -0.024189205354, 0.025960558334),
    Eigen::Vector3d(0.997291125812, 0.034821098222, -0.064791214641),
    Eigen::Vector3d(0.995542300813, -0.052353244304, -0.078451673676),
    Eigen::Vector3d(0.996164275277, -0.016894499435, 0.085856348342),
    Eigen::Vector3d(0.996171168344, 0.087423396183, -0.000391356190),
    Eigen::Vector3d(0.999606603437, -0.026366661016, 0.009562298419),
    Eigen::Vector3d(0.999655700227, -0.012614158014, -0.023007912134),
    Eigen::Vector3d(0.998400624868, -0.023795220252, 0.051283328250),
    Eigen::Vector3d(0.999876330212, -0.007762714527, 0.013677154110),
    Eigen::Vector3d(0.986213169792, 0.089103965374, 0.139441984649),
    Eigen::Vector3d(0.996222059817, 0.025642626265, 0.082970255228),
    Eigen::Vector3d(0.996384866592, -0.025178751513, 0.081137094468),
    Eigen::Vector3d(0.999844157051, -0.014155209844, 0.010549485589),
    Eigen::Vector3d(0.998208655520, 0.027948049772, 0.052899778432),
    Eigen::Vector3d(0.991383552368, 0.019675652476, -0.129504906450),
    Eigen::Vector3d(0.997333312240, -0.066931805634, 0.029092914778),
    Eigen::Vector3d(0.997820936858, -0.063986585051, -0.016096425114),
    Eigen::Vector3d(0.995584753642, 0.087936461301, 0.032835607036),
    Eigen::Vector3d(0.995713593183, -0.078768298496, -0.048476752188),
    Eigen::Vector3d(0.998498491394, 0.006373648865, -0.054407162062),
    Eigen::Vector3d(0.997608275176, -0.049833512845, 0.047899376805),
    Eigen::Vector3d(0.999667070147, -0.002691365376, -0.025661360390),
    Eigen::Vector3d(0.997529213840, 0.007955986133, 0.069800929941),
    Eigen::Vector3d(0.997433810650, 0.044786446784, 0.055856669760),
    Eigen::Vector3d(0.991513206727, 0.124572887587, -0.037190812912),
    Eigen::Vector3d(0.989556687948, -0.060321536899, 0.130915520559),
    Eigen::Vector3d(-0.028457732736, 0.009776999194, 0.999547181345),
    Eigen::Vector3d(0.060655755945, 0.021381253111, 0.997929717608),
    Eigen::Vector3d(0.031126913065, 0.019993373796, 0.999315455843),
    Eigen::Vector3d(-0.050676972591, 0.011263501397, 0.998651579874),
    Eigen::Vector3d(0.004708408275, -0.031789880693, 0.999483483794),
    Eigen::Vector3d(0.078527160719, -0.041858579024, 0.996032802869),
    Eigen::Vector3d(-0.025731637350, 0.009319503092, 0.999625444705),
    Eigen::Vector3d(-0.081998899938, 0.056728329037, 0.995016621516),
    Eigen::Vector3d(-0.059350351697, -0.052424734447, 0.996859660620),
    Eigen::Vector3d(0.025171966620, 0.035765728197, 0.999043134596),
    Eigen::Vector3d(-0.030521319179, 0.049605317053, 0.998302439943),
    Eigen::Vector3d(0.103427315233, -0.030973278532, 0.994154639118),
    Eigen::Vector3d(-0.032466474825, 0.042812111184, 0.998555482258),
    Eigen::Vector3d(-0.026485927520, 0.022217259298, 0.999402265873),
    Eigen::Vector3d(-0.045095641819, 0.016044400650, 0.998853823288),
    Eigen::Vector3d(0.123415347768, 0.015996363688, 0.992226167909),
    Eigen::Vector3d(0.013433663761, 0.038346015524, 0.999174218929),
    Eigen::Vector3d(0.049939648233, -0.015099151712, 0.998638096185),
    Eigen::Vector3d(-0.021428669664, 0.030641376849, 0.999300714571),
    Eigen::Vector3d(-0.043088920446, -0.037743600817, 0.998358034741),
    Eigen::Vector3d(-0.067620541354, -0.050293352500, 0.996442693325),
    Eigen::Vector3d(-0.013600906061, 0.055819698199, 0.998348224142),
    Eigen::Vector3d(0.007299977031, 0.000228055695, 0.999973328807),
    Eigen::Vector3d(0.126746909209, -0.058691354316, 0.990197225776),
    Eigen::Vector3d(-0.034104034197, 0.011765402081, 0.999349033204),
    Eigen::Vector3d(-0.067602154578, 0.074811492226, 0.994903608058),
    Eigen::Vector3d(-0.057963761066, 0.040528620190, 0.997495680867),
    Eigen::Vector3d(-0.056332209381, -0.012680233559, 0.998331555077),
    Eigen::Vector3d(0.044636511766, -0.015127140601, 0.998888758288),
    Eigen::Vector3d(-0.019097128647, 0.001644921629, 0.999816280079),
    Eigen::Vector3d(0.017841577477, 0.077097940743, 0.996863875184),
    Eigen::Vector3d(0.003246663501, -0.002232376590, 0.999992237805),
    Eigen::Vector3d(-0.022537957702, 0.009438650150, 0.999701431601),
    Eigen::Vector3d(0.003937175678, 0.038481279607, 0.999251564806),
    Eigen::Vector3d(-0.047636842396, -0.053157384637, 0.997449258712),
    Eigen::Vector3d(-0.081097713066, 0.007979598779, 0.996674213040),
    Eigen::Vector3d(0.046705634917, 0.101558378448, 0.993732599563),
    Eigen::Vector3d(-0.023249367001, 0.022859480354, 0.999468314201),
    Eigen::Vector3d(0.019589762514, -0.034651325069, 0.999207449370),
    Eigen::Vector3d(0.055113706602, -0.014048610838, 0.998381247760)};

const std::vector<double> weights_squared{
    0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000,
    0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000,
    0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000,
    0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000,
    0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000,
    0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000,
    0.001000000000, 0.001000000000, 0.001000000000, 0.001000000000, 0.246642904145, 0.479711029065,
    0.242397120757, 0.977327132907, 0.781277550151, 0.931103781143, 0.092233253659, 0.341685443882,
    0.156684896784, 0.269451407514, 0.934844360520, 0.069635843998, 0.813079271404, 0.879006709745,
    0.574432348813, 0.303442251759, 0.256489716159, 0.243684836021, 0.312260595383, 0.134478419599,
    0.583301652536, 0.579465212921, 0.697629018453, 0.391440985668, 0.264070085818, 0.693095706697,
    0.926364441024, 0.922090580363, 0.222589560573, 0.297704203720, 0.519670631483, 0.863039524950,
    0.230692125873, 0.801459091744, 0.062389197679, 0.517996619843, 0.260165296287, 0.482540955893,
    0.064965226936, 0.113423356514, 0.917451750123, 0.844864931393, 0.680701080864, 0.129504851961,
    0.338351889484, 0.168437209969, 0.039085128019, 0.755524228624, 0.624238900159, 0.900517619013,
    0.728937054568, 0.657606045364, 0.852281778254, 0.692405258748, 0.559158979315, 0.315175869792,
    0.397143102569, 0.476813452220, 0.783373591554, 0.911610677516, 0.488686481564, 0.397166613515,
    0.123634133266, 0.144661877178, 0.610563804627, 0.355478768579, 0.966750021854, 0.262735422630,
    0.286972432213, 0.825419052651, 0.696510514691, 0.439465101430, 0.168204790000, 0.535144657354,
    0.113530775420, 0.653672369920, 0.422294261891, 0.452069867538, 0.794739889573, 0.136847680940};

// Variances when using the identy matrix as a dummy for the eigenvectors
const std::vector<double> variances{857.604389492433, 835.286401424455, 1328.037416955534,
                                    0.000982026674, 0.000777585317, 0.000896379977};

// Probabilities when using the identy matrix as a dummy for the eigenvectors
const std::vector<double> probabilities{0.999999999997, 0.999999285813, 0.998666038551,
                                        1.000000000000, 0.000848760948, 1.000000000000};

const std::vector<double> noise_expectation_matrix_entries{
    7.161824895883, -0.539247129834, 0.497459842828, 0.002069293271, 0.049940005213,
    -0.087520650244, -0.539247129834, 10.558186149572, 0.795316850034, -0.051871704170,
    0.002829984832, -0.010786377197, 0.497459842828, 0.795316850034, 16.424262559366,
    -0.096069776189, 0.113820522616, -0.004899278102, 0.002069293271, -0.051871704170,
    -0.096069776189, 0.051816914278, -0.000186003960, -0.000215326530, 0.049940005213,
    0.002829984832, 0.113820522616, -0.000186003960, 0.097772318174, -0.000774598260,
    -0.087520650244, -0.010786377197, -0.004899278102, -0.000215326530, -0.000774598260,
    0.046519238872};

const std::vector<double> measured_info_matrix_entries{
    2297.775050794187, -386.126320724752, 16.078291747225, -0.827717308224, 0.849226730183,
    38.519513812148, -386.126320724752, 1648.807319509154, -31.903722381034, -0.076547147473,
    -1.131993932669, -42.568433939491, 16.078291747225, -31.903722381034, 1384.710375115361,
    34.916656761404, 1.354775771654, 1.959711240894, -0.827717308224, -0.076547147473,
    34.916656761404, 18.494928553410, 0.074401583962, 0.086130612182, 0.849226730183,
    -1.131993932669, 1.354775771654, 0.074401583962, 0.112766995270, 0.309839303808,
    38.519513812148, -42.568433939491, 1.959711240894, 0.086130612182, 0.309839303808,
    20.613998716046};

const double stdev_points = 0.100000000000;
const double stdev_normals = 0.050000000000;

}  // namespace data