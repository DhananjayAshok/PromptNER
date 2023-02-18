#!/usr/bin/perl
require 5.000;

use Getopt::Std;
getopts('ha:t:');

$SS_INIT = "~$~";
$SS_WILD = "~*~";

if (($opt_h) || ($#ARGV == -1)) {
    die "\n[evalIOB2] 2004.4.2 Jin-Dong Kim (mail\@jdkim.net)\n" .
	"\n<DESCRIPTION>\n" .
	"\nIt evaluates the performance of object identification in terms of precision and recall." .
	"\nIt assumes the object identification is encoded in the IOB2 tagging scheme.\n" .
	"\n<USAGE>\n" .
	"\nevalIOB2.pl [-h] [-r ref_field] [-a answer_field] ref_file [answer_file]\n" .
	"\n<OPTIONS>\n" .
	"\n-h               shows this instructions.\n" .
	"\n-r ref_field     specifies the location of the reference field in the reference file." .
	"\n                 It is 0-oriented and the default is -1 (the last field)," .
	"\n                 or -2 (the second to last field) when the answer file is not specified.\n" .
	"\n-a answer_field  specifies the location of the answer field in the answer file" .
	"\n                 or in the reference file when the answer file is not specified." .
	"\n                 the default is -1 in any case.\n\n";
} # if

if ($#ARGV == 1) {
    open (RFILE, $ARGV[0]) or die "can't open [$ARGV[0]].\n";
    open (AFILE, $ARGV[1]) or die "can't open [$ARGV[1]].\n";
    $rfield = $afield = -1;
} # if

elsif ($#ARGV == 0) {
    open (RFILE, $ARGV[0]) or die "can't open [$ARGV[0]].\n";
    open (AFILE, $ARGV[0]) or die "can't open [$ARGV[0]].\n";
    $rfield = -2;
    $afield = -1;
} # elsif

if (defined($opt_r)) {$rfield = $opt_r}
if (defined($opt_a)) {$afield = $opt_a}
if (defined($opt_l)) {$lfield = $opt_l}

$numtag=$numctag=$numbtag=0;
$numans=$numref=$numcrt=$numleft=$numright=0;
$numbcrt=$numbleft=$numbright=0;

while (@ablock=&read_block(\*AFILE)) {

    @tags = ();
    foreach $token (@ablock) {push(@tags, ${$token}[$afield])}
    @ntags = &iob2_iobes(@tags);
    for ($i=0; $i<=$#ntags; $i++) {${$ablock[$i]}[$afield] = $ntags[$i]}

    @rblock=&read_block(\*RFILE);

    if ($#ablock != $#rblock) {die "the number of tokens in a sentence is different.\n"}

    @tags = ();
    foreach $token (@rblock) {push(@tags, ${$token}[$rfield])}
    @ntags = &iob2_iobes(@tags);
    for ($i=0; $i<=$#ntags; $i++) {${$rblock[$i]}[$rfield] = $ntags[$i]}

    $match=0; $bmatch=0;
    for ($i=0; $i<=$#ablock; $i++) {

	$atag = ${$ablock[$i]}[$afield];
	$aiob = substr($atag, 0, 1);
	if (substr($atag, 1, 1) eq "-") {$acls = substr($atag, 2)}
	else {$acls = ""}

	$rtag = ${$rblock[$i]}[$rfield];
	$riob = substr($rtag, 0, 1);
	if (substr($rtag, 1, 1) eq "-") {$rcls = substr($rtag, 2)}
	else {$rcls = ""}

	$numtag++;
	if ($aiob eq $riob) {$numbtag++}
	if ($atag eq $rtag) {$numctag++}

	if (($riob eq "S") || ($riob eq "B")) {
	    if (defined($lfield)) {
		@lexs = split ' ', ${$rblock[$i]}[$lfield];
		$numlexs = $#lexs+1;
	    } # if
	    else {
		$numlexs = 1;
	    } # else
	    &cntref;
	} # if

	if (($aiob eq "S") || ($aiob eq "B")) {&cntans("numans")}

	if (($aiob eq "S") && ($riob eq "S")) {
	    &cntans("numbcrt"); &cntans("numbleft"); &cntans("numbright");
	    if ($acls eq $rcls) {&cntans("numcrt"); &cntans("numleft"); &cntans("numright")}
	} # if

	if (($aiob eq "S") && ($riob eq "E")) {
	    &cntans("numbright");
	    if ($acls eq $rcls) {&cntans("numright")}
	} # if

	if (($aiob eq "S") && ($riob eq "B")) {
	    &cntans("numbleft");
	    if ($acls eq $rcls) {&cntans("numleft")}
	} # if

	if (($aiob eq "B") && ($riob eq "S")) {
	    &cntans("numbleft");
	    if ($acls eq $rcls) {&cntans("numleft")}
	} # if

	if (($aiob eq "B") && ($riob eq "B")) {
	    &cntans("numbleft"); $bmatch = 1;
	    if ($acls eq $rcls) {&cntans("numleft"); $match = 1}
	} # if


	if ($aiob ne $riob) {$bmatch=0}
	if ($atag ne $rtag) {$match=0}


	if (($aiob eq "O") || ($riob eq "O")) {$match=0; $bmatch=0}

	if (($aiob eq "E") && ($riob eq "S")) {
	    &cntans("numbright");
	    if ($acls eq $rcls) {&cntans("numright")}
	} # if


	if (($aiob eq "E") && ($riob eq "E")) {
	    if ($bmatch) {&cntans("numbcrt")}
	    &cntans("numbright");
	    if ($acls eq $rcls) {if ($match) {&cntans("numcrt")}; &cntans("numright");}
	} # if

    } # for ($i)

} # while

print "\n[Tagging Performance]\n";
print "# of tags: $numtag,\t correct tags: $numctag,\t correct IOBs: $numbtag\n";
printf ("precision with class info: %6.4f,\t w/o class info: %6.4f\n", $numctag/$numtag, $numbtag/$numtag);

print "\n[Object Identification Performance]\n";
print "# of OBJECTs: $numref,\t ANSWERs: $numans.\n";
print "\n# (recall / precision / f-score) of ...\n";

if ($numref == 0) {die "[!]No object to identify.\n"}
if ($numans == 0) {die "[!]No object identified.\n"}

$recall=$numcrt/$numref;
$precision=$numcrt/$numans;
if ($precision+$recall==0) {$fscore = 0} else {$fscore=2*$precision*$recall/($precision+$recall)}
printf ("     FULLY CORRECT answer with class info: $numcrt (%6.4f / %6.4f / %6.4f),\n", $recall, $precision, $fscore);

$recall=$numbcrt/$numref;
$precision=$numbcrt/$numans;
if ($precision+$recall==0) {$fscore = 0} else {$fscore=2*$precision*$recall/($precision+$recall)}
#printf ("                           w/o class info: $numbcrt (%6.4f / %6.4f / %6.4f),\n", $recall, $precision, $fscore);

$recall=$numleft/$numref;
$precision=$numleft/$numans;
if ($precision+$recall==0) {$fscore = 0} else {$fscore=2*$precision*$recall/($precision+$recall)}
printf ("    correct LEFT boundary with class info: $numleft (%6.4f / %6.4f / %6.4f),\n", $recall, $precision, $fscore);

$recall=$numbleft/$numref;
$precision=$numbleft/$numans;
if ($precision+$recall==0) {$fscore = 0} else {$fscore=2*$precision*$recall/($precision+$recall)}
#printf ("                           w/o class info: $numbleft (%6.4f / %6.4f / %6.4f),\n", $recall, $precision, $fscore);

$recall=$numright/$numref;
$precision=$numright/$numans;
if ($precision+$recall==0) {$fscore = 0} else {$fscore=2*$precision*$recall/($precision+$recall)}
printf ("   correct RIGHT boundary with class info: $numright (%6.4f / %6.4f / %6.4f),\n", $recall, $precision, $fscore);

$recall=$numbright/$numref;
$precision=$numbright/$numans;
if ($precision+$recall==0) {$fscore = 0} else {$fscore=2*$precision*$recall/($precision+$recall)}
#printf ("                           w/o class info: $numbright (%6.4f / %6.4f / %6.4f).\n", $recall, $precision, $fscore);
printf "\n";


if (keys(%numref) > 1) {

foreach $obj (keys(%numref)) {

if (!defined($numans{$obj})) {$numans{$obj}=0}
if (!defined($numcrt{$obj})) {$numcrt{$obj}=0}
if (!defined($numbcrt{$obj})) {$numbcrt{$obj}=0}
if (!defined($numleft{$obj})) {$numleft{$obj}=0}
if (!defined($numbleft{$obj})) {$numbleft{$obj}=0}
if (!defined($numright{$obj})) {$numright{$obj}=0}
if (!defined($numbright{$obj})) {$numbright{$obj}=0}

printf ("\n[<%s> Identification Performance]\n", $obj);
print "# of OBJECTs: $numref{$obj},\t ANSWERs: $numans{$obj}.\n";
print "\n# (recall / precision / f-score) of ...\n";

if ($numref{$obj}==0) {$recall=0} else{$recall=$numcrt{$obj}/$numref{$obj}}
if ($numans{$obj}==0) {$precision=0} else{$precision=$numcrt{$obj}/$numans{$obj}}
if ($precision+$recall==0) {$fscore = 0} else {$fscore=2*$precision*$recall/($precision+$recall)}
printf ("     FULLY CORRECT answer with class info: $numcrt{$obj} (%6.4f / %6.4f / %6.4f),\n", $recall, $precision, $fscore);

#if ($numref{$obj}==0) {$recall=0} else{$recall=$numbcrt{$obj}/$numref{$obj}}
#if ($numans{$obj}==0) {$precision=0} else{$precision=$numbcrt{$obj}/$numans{$obj}}
#if ($precision+$recall==0) {$fscore = 0} else {$fscore=2*$precision*$recall/($precision+$recall)}
#printf ("                           w/o class info: $numbcrt{$obj} (%6.4f / %6.4f / %6.4f),\n", $recall, $precision, $fscore);

if ($numref{$obj}==0) {$recall=0} else{$recall=$numleft{$obj}/$numref{$obj}}
if ($numans{$obj}==0) {$precision=0} else{$precision=$numleft{$obj}/$numans{$obj}}
if ($precision+$recall==0) {$fscore = 0} else {$fscore=2*$precision*$recall/($precision+$recall)}
printf ("    correct LEFT boundary with class info: $numleft{$obj} (%6.4f / %6.4f / %6.4f),\n", $recall, $precision, $fscore);

#if ($numref{$obj}==0) {$recall=0} else{$recall=$numbleft{$obj}/$numref{$obj}}
#if ($numans{$obj}==0) {$precision=0} else{$precision=$numbleft{$obj}/$numans{$obj}}
#if ($precision+$recall==0) {$fscore = 0} else {$fscore=2*$precision*$recall/($precision+$recall)}
#printf ("                           w/o class info: $numbleft{$obj} (%6.4f / %6.4f / %6.4f),\n", $recall, $precision, $fscore);

if ($numref{$obj}==0) {$recall=0} else{$recall=$numright{$obj}/$numref{$obj}}
if ($numans{$obj}==0) {$precision=0} else{$precision=$numright{$obj}/$numans{$obj}}
if ($precision+$recall==0) {$fscore = 0} else {$fscore=2*$precision*$recall/($precision+$recall)}
printf ("   correct RIGHT boundary with class info: $numright{$obj} (%6.4f / %6.4f / %6.4f),\n", $recall, $precision, $fscore);

#if ($numref{$obj}==0) {$recall=0} else{$recall=$numbright{$obj}/$numref{$obj}}
#if ($numans{$obj}==0) {$precision=0} else{$precision=$numbright{$obj}/$numans{$obj}}
#if ($precision+$recall==0) {$fscore = 0} else {$fscore=2*$precision*$recall/($precision+$recall)}
#printf ("                           w/o class info: $numbright{$obj} (%6.4f / %6.4f / %6.4f).\n", $recall, $precision, $fscore);
printf "\n";

} # foreach

} # if

sub cntans {
    my $cntname = shift(@_);
    $$cntname += $numlexs;
    if (defined($cntname->{$acls})) {$cntname->{$acls} += $numlexs}
    else {$cntname->{$acls} = $numlexs}
} # cntans

sub cntref {
    $numref += $numlexs;
    if (defined($numref{$rcls})) {$numref{$rcls} += $numlexs}
    else {$numref{$rcls} = $numlexs}
} # cntref

sub read_block {
    my ($FILE) = shift;
    my (@block);

    while (<$FILE>) {
	chomp;
	if (blank_line($_)) {
	    if (@block) {last;}
	    else {next;}
	} # if
	push(@block, [split(/\t/, $_)]);
    } # while

    return @block;
} # read_block


sub blank_line {
    my $line = shift(@_);
    return (($line eq "") || ($line =~ /^\#\#\#MEDLIN/));
} # blank_line


sub iob2_iobes (@tags) {
    my (@tags) = @_;
    my (@ntags, $i);

    for ($i=0; $i<=$#tags; $i++) {
	$ntags[$i] = $tags[$i];

	if (substr($tags[$i], 0, 1) eq "I") {

	    if (($i==$#tags)||($tags[$i+1] eq $SS_INIT)
		||(substr($tags[$i+1], 0, 1) ne "I")) {substr($ntags[$i], 0, 1) = "E"}
	    else {substr($ntags[$i], 0, 1) = "I"}

	} elsif (substr($tags[$i], 0, 1) eq "B") {

	    if (($i==$#tags)||($tags[$i+1] eq $SS_INIT)
		||(substr($tags[$i+1], 0, 1) ne "I")) {substr($ntags[$i], 0, 1) = "S"}
	    else {substr($ntags[$i], 0, 1) = "B"}

	} elsif (substr($tags[$i], 0, 1) eq "O") {

	    substr($ntags[$i], 0, 1) = "O";

	} # else
    } # for

    return @ntags;
} # iob2_iobes
